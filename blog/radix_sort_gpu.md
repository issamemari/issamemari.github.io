# Implementing Radix Select for the GPU: A Worklog

*January 2026*

In this post, I'll implement a radix-based selection algorithm for GPUs from scratch. My goal is to build a TopK kernel that's actually competitive with the state of the art—which, as I discovered the hard way, is a higher bar than I expected.

## The Humbling

I started this project thinking I had a decent TopK implementation. I was combining ideas from [Dr. TopK](https://arxiv.org/abs/2104.13519)^[Dr. TopK splits the array into ranges, finds delegates (maximum per range), keeps the TopK among the delegates, removes ranges whose delegate isn't in the TopK, then concatenates and runs TopK on what remains.] with a simple selection approach: loop `i = 0` to `K`, find the maximum in `array[i:]`, swap it with `array[i]`. After some optimizations—parallel reductions, warp-level operations, block-level coordination—I thought I had myself a pretty good kernel.

Then I benchmarked against Thrust.

```
./dr_topk_parallel_topk_unroll_warp_parallel_block_reduction_param_optimized 100000000 1000
Running dr_topk with N=100000000, K=1000
=== Algorithm Performance Report ===
N = 100000000, K = 1000

Step 1 - Data preparation & H2D copy: 5554.255 ms
Step 2 - Find delegates (RW=4096, threads=24415): 31.818 ms
Step 3 - Prepare delegate indices & H2D copy: 0.231 ms
Step 4 - Find top-K from delegates: 19.117 ms
Step 5 - Concatenate top-K ranges: 0.246 ms
Step 6 - Final top-K on concatenated ranges: 282.780 ms
Step 7 - D2H copy: 0.082 ms
```

The total kernel time is around 334ms. Now compare to Thrust just... sorting the entire array:

```
./thrust_topk 100000000 1000
=== Thrust Sort TopK ===
N = 100000000, K = 1000

Step 1 - Data preparation & H2D copy: 5380.762 ms
Step 2 - Thrust sort (descending): 39.574 ms
Step 3 - D2H copy: 0.043 ms
```

**39ms.** Thrust can *sort 100 million elements* faster than my kernel can find the top 1000. And my kernel should be faster—I'm doing less work! I only need to find K items, not establish a total ordering.

This was deflating. But also clarifying: my baseline should be sorting with Thrust and taking the first K. If I want to beat that, I need to use the same underlying technique Thrust uses. I dug into the [Thrust source code](https://github.com/nvidia/thrust) and found the answer: radix sort.

## A Quick Primer on Radix Sort

Radix sort is one of those algorithms that seems almost too simple to work, and yet it's the foundation of the fastest GPU sorting implementations.^[It's a non-comparison sort, which lets it break the `O(n log n)` lower bound for comparison-based sorts. The complexity is `O(nw)` where `w` is the number of digits/bits.]

The basic idea: repeatedly bucket elements by their digits, starting from the least significant. Here's a worked example with decimal numbers:

```
array: [23, 66, 12, 539, 12, 32, 61]
```

**Pass 1: Bucket by ones digit**
```
bucket 0: 
bucket 1: 61
bucket 2: 12, 12, 32
bucket 3: 23
bucket 6: 66
bucket 9: 539
```

Unload in bucket order: `[61, 12, 12, 32, 23, 66, 539]`

**Pass 2: Bucket by tens digit**
```
bucket 1: 12, 12
bucket 2: 23
bucket 3: 32, 539
bucket 6: 61, 66
```

Unload: `[12, 12, 23, 32, 539, 61, 66]`

**Pass 3: Bucket by hundreds digit**
```
bucket 0: 12, 12, 23, 32, 61, 66
bucket 5: 539
```

Unload: `[12, 12, 23, 32, 61, 66, 539]` ✓

That's it. The array is sorted. The magic is that because we process least-significant digits first and the bucketing is stable (elements maintain their relative order within buckets), each pass refines the previous without destroying it.

In practice, GPU implementations work on binary and process multiple bits at once—typically 4-8 bits per pass, giving 16-256 buckets. But the principle is identical.

## From Sorting to Selection

Here's where it gets interesting. We don't need to sort. We just need the top K.

Consider the same array, but now we want the top 3 elements. Instead of processing least-significant bits first, let's start from the *most* significant digit:

```
array: [23, 66, 12, 539, 12, 32, 61]
K = 3
```

**Pass 1: Bucket by hundreds digit**
```
bucket 0: 23, 66, 12, 12, 32, 61  (6 elements)
bucket 5: 539                     (1 element)
```

Stop and think. The bucket 5 contains 1 element. Since we want the top 3, and bucket 5 is the highest bucket, all of its elements are *guaranteed* to be in the top K. We've found one answer: **539**.

Now we need 2 more elements from bucket 0.

**Pass 2: Bucket the remaining elements by tens digit**
```
bucket 1: 12, 12
bucket 2: 23
bucket 3: 32
bucket 6: 66, 61
```

Again, work from the top. Bucket 6 has 2 elements. Combined with the 1 we already found, that's exactly 3. Done.

**Top 3: [539, 66, 61]** ✓

The key insight: at each digit position, we can immediately identify elements that *must* be in the top K (buckets at the high end whose cumulative count from the top doesn't exceed K) and elements that *can't* be (buckets at the low end whose cumulative count from the bottom exceeds N-K). We only recurse into the *pivot bucket*—the one that straddles the boundary.

## The Algorithm, Formally

Here's a Python implementation that captures the recursive structure:

```python
def radix_select(array, k, digit_rank) -> list:
    if digit_rank < 0:
        return list(array)[:k]

    if len(array) == 0:
        return []

    if len(array) <= k:
        return list(array)

    # Bucket by current digit
    buckets = [[] for _ in range(10)]
    for number in array:
        digit = (number // (10**digit_rank)) % 10
        buckets[digit].append(number)

    # Scan from highest bucket downward
    found = []
    for i in range(len(buckets))[::-1]:
        if len(buckets[i]) == 0:
            continue

        if len(found) + len(buckets[i]) <= k:
            # This entire bucket is in top-K
            found.extend(buckets[i])
            if len(found) == k:
                return found
        else:
            # This bucket straddles the boundary—recurse
            needed = k - len(found)
            return found + radix_select(buckets[i], needed, digit_rank - 1)

    return found
```


## From Python to CUDA: The Practical Challenges

Before we can implement this on a GPU, we need to address some impedance mismatches between the Python sketch and CUDA C.

### Dynamic Allocation is the Enemy

The Python implementation casually creates new lists for each bucket at each recursion level. This is fine for a 20-line prototype, but on a GPU, dynamic memory allocation is expensive—and doing it from within a kernel is even worse.^[You *can* use `malloc` from device code (since compute capability 2.x), but it's backed by a fixed-size heap and has terrible performance characteristics for fine-grained allocations.]

The fix is a classic trick: **pre-allocate once, partition logically**. We allocate a single output array the same size as the input, then use index arithmetic to carve it into buckets.

The implementation requires three passes:

1. **Histogram pass**: Count how many elements fall into each bucket
2. **Prefix sum pass**: Compute prefix sums to get bucket offsets
3. **Scatter pass**: Write each element to its destination
```
Input:  [23, 66, 12, 539, 12, 32, 61]

Pass 1 - Count by hundreds digit:
  bucket 0: 6 elements
  bucket 5: 1 element

Pass 2 - Compute offsets via prefix sum:
  bucket 0 starts at index 0
  bucket 5 starts at index 6

Pass 3 - Scatter to output:
  Output: [23, 66, 12, 12, 32, 61, 539]
           └─── bucket 0 ───┘   └─ bucket 5
```

Each thread reads one element, determines its bucket, atomically increments a per-bucket counter to claim a slot, and writes to that slot. The atomic increment is the only synchronization needed.^[This is still a simplification. Real implementations use per-block histograms to reduce contention, then combine them. We'll get there.]

### The Negative Number Problem

There's a more subtle issue lurking in our algorithm. Let's trace through radix select with some negative numbers:
```
array: [23, -66, 12, -539, 12, 32, -61]
K = 3 (we want the three largest: 32, 23, 12)
```

Our algorithm buckets by the most significant digit first. For the hundreds place:
```
digit 0: 23, 12, 12, 32   (4 elements)
digit 5: -539             (1 element)  
```

Wait—where do -66 and -61 go? They don't have a hundreds digit in the same sense. And -539: is its hundreds digit 5, or is it somehow "negative 5"?

Let's try a simpler example. Consider sorting these numbers in descending order:
```
array: [5, -3, 2, -1, 4]
```

The correct descending order is: **[5, 4, 2, -1, -3]**

If we bucket by the ones digit (ignoring the sign for now):
```
digit 1: -1
digit 2: 2
digit 3: -3
digit 4: 4
digit 5: 5
```

Scanning from highest bucket down for K=2, we'd pick 5 and 4. That happens to be correct! But we got lucky—let's try K=3:

Scanning down: digit 5 gives us 5, digit 4 gives us 4, digit 3 gives us... **-3**.

We'd return **[5, 4, -3]**, but the correct answer is **[5, 4, 2]**.

The problem is fundamental: our algorithm assumes that higher digits mean larger values. But -3 isn't larger than 2 just because 3 > 2. The sign completely changes the ordering.

#### What We Need

For radix select to work, we need a representation where the digit-by-digit comparison matches numeric ordering. Specifically:

1. All positive numbers should sort higher than all negative numbers
2. Among positives, larger magnitude = larger value (this already works)
3. Among negatives, larger magnitude = *smaller* value (-1 > -100)

Decimal digits can't give us this directly. We need to transform the numbers into a representation where "lexicographically larger" equals "numerically larger."

This is where we'll need to dive into how computers actually represent negative numbers—using a system called two's complement. The good news: there's a simple bit-manipulation trick that makes everything work. But first, let's understand the representation.

#### A Primer on Two's Complement

Let's work with 8-bit integers first—the principles scale to 32 and 64 bits.

An **unsigned 8-bit integer** (`uint8`) uses all 8 bits to represent magnitude. The range is 0 to 255:
```
0   = 0b00000000
1   = 0b00000001
127 = 0b01111111
128 = 0b10000000
255 = 0b11111111
```

Simple enough: the bit pattern *is* the number in binary. Larger values mean larger bit patterns, and vice versa. Radix sort on unsigned integers just works.

For **signed 8-bit integers** (`int8`), we need to represent negative numbers somehow. The nearly universal solution is **two's complement**. The range is -128 to 127:
```
 127 = 0b01111111
 126 = 0b01111110
   1 = 0b00000001
   0 = 0b00000000
  -1 = 0b11111111
  -2 = 0b11111110
-127 = 0b10000001
-128 = 0b10000000
```

Notice the pattern:
- **Positive numbers** (0 to 127): MSB is 0, remaining bits are the magnitude. Same as unsigned.
- **Negative numbers** (-1 to -128): MSB is 1. To get the magnitude, flip all bits and add 1.^[This is why it's called *two's complement*—the negative of a number is computed by taking the bitwise complement and adding one. Equivalently: `-x = ~x + 1`.]

Let's verify with -1:
```
-1 in two's complement: 0b11111111
Flip all bits:          0b00000000
Add 1:                  0b00000001 = 1 ✓
```

And -128:
```
-128 in two's complement: 0b10000000
Flip all bits:            0b01111111
Add 1:                    0b10000000 = 128 ✓
```

Two's complement has a beautiful property: addition *just works* without special-casing the sign. The hardware doesn't need separate circuits for signed vs unsigned addition.^[This is why two's complement won the representation wars. One's complement and sign-magnitude both require special handling for arithmetic operations.]

#### Why This Breaks Radix Select

Back to our example, now with 8-bit two's complement:
```
array: [5, -3, 2, -1, 4]
K = 2 (we want the two largest: 5 and 4)
```

The bit representations:
```
  5 = 0b00000101
  4 = 0b00000100
  2 = 0b00000010
 -1 = 0b11111111
 -3 = 0b11111101
```

Our algorithm buckets by most significant bits first. With a 2-bit radix (4 buckets), looking at bits 7-6:
```
Bucket 0b00 (0): 5, 4, 2  (all positives)
Bucket 0b01 (1): (empty)
Bucket 0b10 (2): (empty)
Bucket 0b11 (3): -1, -3   (all negatives)
```

The algorithm scans from the highest bucket down. Bucket 3 has 2 elements—exactly K! So it returns **[-1, -3]** as the top 2.

But the actual top 2 are **[5, 4]**. We're completely wrong.

The root cause: when interpreted as *unsigned* integers, the bit patterns of negative numbers are larger than those of positive numbers:
```
Signed → Unsigned interpretation
  5 (0b00000101) →   5
  4 (0b00000100) →   4
  2 (0b00000010) →   2
 -1 (0b11111111) → 255
 -3 (0b11111101) → 253
```

Radix sort/select operates on bit patterns, not semantic values. It sees 255 and 253 as the largest numbers.

To make this concrete, here's how numeric ordering compares to bit-pattern ordering:
```
Numeric order (signed):    -128 < -1 < 0 < 1 < 127
Bit-pattern order (uint):   0 < 1 < 127 < 128 < 255
                            ↑   ↑    ↑     ↑     ↑
Actual signed values:       0   1  127  -128   -1
```

The positive numbers are fine—their relative order is preserved. But negatives come *after* all positives in bit-pattern order, and they're internally reversed: -1 (0xFF) appears larger than -128 (0x80).

#### The Fix: Flip the Sign Bit

We need a transformation that makes bit-pattern ordering match numeric ordering. The solution is surprisingly simple: **flip the most significant bit**.
```
Original → Flip MSB
  5 (0b00000101) → 0b10000101 (133)
  4 (0b00000100) → 0b10000100 (132)
  2 (0b00000010) → 0b10000010 (130)
 -1 (0b11111111) → 0b01111111 (127)
 -3 (0b11111101) → 0b01111101 (125)
```

Now let's check the ordering of the transformed values:
```
Transformed: 125 < 127 < 130 < 132 < 133
Original:     -3   -1     2     4     5  ✓
```

It works. The transformation preserves numeric ordering while giving us unsigned bit patterns we can radix-sort. Here's why:

1. **Positive numbers** originally have MSB=0. Flipping makes MSB=1, so they all move to the upper half of the unsigned range (128-255 for uint8).

2. **Negative numbers** originally have MSB=1. Flipping makes MSB=0, so they all move to the lower half (0-127 for uint8).

3. **Within positive numbers**, the relative ordering is unchanged—we only flipped the top bit, which was the same (0) for all of them.

4. **Within negative numbers**, the relative ordering is also unchanged—same logic.

5. **Across the boundary**, all (transformed) positives are now greater than all (transformed) negatives, which matches numeric ordering.

For 32-bit integers:
```c
// Transform signed int to radix-sortable unsigned
__device__ __host__ unsigned int to_sortable(int x) {
    return (unsigned int)x ^ 0x80000000u;  // flip MSB
}

// Transform back
__device__ __host__ int from_sortable(unsigned int x) {
    return (int)(x ^ 0x80000000u);  // flip MSB again (self-inverse)
}
```

The `0x80000000u` constant is just a 32-bit value with only the MSB set. XOR with this flips exactly that bit.

We apply `to_sortable()` before radix select, run the algorithm on unsigned values, then apply `from_sortable()` to the results. The transformation is its own inverse, so we use the same XOR operation both ways.

#### What About Floating-Point Numbers?

The same principle applies to floats, but IEEE 754 floating-point representation requires a slightly different transformation. Before we get there, let's understand how floats work.

##### The Problem

Writing integers in binary is straightforward—two's complement handles both positive and negative values elegantly. Floating-point numbers are trickier. The core challenge is representing the decimal point. When designing a binary representation for floats, you need to balance several concerns:

- **Range**: Can you represent both very large and very small numbers?
- **Precision**: How many significant digits can you store?
- **Efficiency**: Can hardware compare and operate on these values quickly?
- **Uniqueness**: Ideally, each number should have exactly one representation.

##### Scientific Notation to the Rescue

The IEEE 754 designers used scientific notation. You're probably familiar with base-10 scientific notation, where you factor a number into a value between 1 and 10, times a power of 10:

$$6500 = 6.5 \times 10^3$$
$$0.0042 = 4.2 \times 10^{-3}$$

Binary scientific notation works the same way, but with powers of 2. Any number can be written as a value between 1 and 2, times a power of 2:

$$6.5 = 1.625 \times 2^2$$
$$0.375 = 1.5 \times 2^{-2}$$

The key insight: if we always normalize so the leading digit is 1, we don't need to store it! In binary, there's only one non-zero digit (1), so every normalized number looks like $1.\text{something} \times 2^p$. We can store just the "something" (the *fraction*) and the power $p$.

##### The Three Pieces

This gives us three components to store:

1. **Sign**: Is the number positive or negative?
2. **Fraction**: The bits after the implicit leading 1
3. **Exponent**: The power of 2

For a 32-bit float, these are packed as:
```
[S][EEEEEEEE][FFFFFFFFFFFFFFFFFFFFFFF]
 1     8              23 bits
 ↑     ↑               ↑
sign exponent       fraction
```

**The sign bit** is simple: 0 for positive, 1 for negative.

**The fraction** represents the digits after the binary point. Each bit position corresponds to a negative power of 2:

| Bit position | $22$ | $21$ | $20$ | $19$ | ... | $0$ |
|--------------|----|----|----|----|-----|---|
| Value | $2^{-1}$ | $2^{-2}$ | $2^{-3}$ | $2^{-4}$ | ... | $2^{-23}$ |
| Decimal | $0.5$ | $0.25$ | $0.125$ | $0.0625$ | ... | $~0.00000012$ |

For example, the fraction `101` (followed by 20 zeros) means $2^{-1} + 2^{-3} = 0.5 + 0.125 = 0.625$. With the implicit leading 1, the full significand is $1.625$.

**The exponent** needs to represent both positive powers (for large numbers) and negative powers (for small numbers). Rather than using two's complement, IEEE 754 uses a *biased* representation: add 127 to the actual exponent before storing it. This converts the range $[-126, +127]$ into the unsigned range $[1, 254]$.^[The values 0 and 255 are reserved for special cases: zero, subnormal numbers, infinity, and NaN.]

Why bias instead of two's complement? Because it makes comparison easier. With biased exponents, larger stored values mean larger actual exponents—no sign bit to worry about.

##### Putting It Together

The formula to decode a 32-bit float:

$$\text{value} = (-1)^S \times (1 + F) \times 2^{E-127}$$

where $S$ is the sign bit, $F$ is the fraction interpreted as described above, and $E$ is the stored (biased) exponent.

Let's encode $6.5$:

1. Convert to binary: $6.5 = 110.1_2$
2. Normalize: $110.1_2 = 1.101_2 \times 2^2$
3. Extract the pieces:
   - Sign = 0 (positive)
   - Fraction = `10100000000000000000000` (the `.101` after the implicit 1)
   - Exponent = $2 + 127 = 129 = 10000001_2$

Result: `0|10000001|10100000000000000000000` = 0x40D00000

##### Why This Layout Enables Fast Comparison

Here's the clever part. The fields are arranged with the sign bit first, then the exponent, then the fraction. For positive floats, this means the bit pattern—interpreted as an unsigned integer—has the same ordering as the numeric value.

**Same exponent, increasing fraction:**
```
Value   S  Exponent   Fraction                   Hex        Unsigned
1.0     0  01111111   00000000000000000000000    0x3F800000  1065353216
1.25    0  01111111   01000000000000000000000    0x3FA00000  1067450368
1.5     0  01111111   10000000000000000000000    0x3FC00000  1069547520
1.75    0  01111111   11000000000000000000000    0x3FE00000  1071644672
```

All four have the same exponent. Larger fraction → larger numeric value → larger bit pattern. The ordering matches.

**Same fraction, increasing exponent:**
```
Value   S  Exponent   Fraction                   Hex        Unsigned
1.5     0  01111111   10000000000000000000000    0x3FC00000  1069547520
3.0     0  10000000   10000000000000000000000    0x40400000  1078001664
6.0     0  10000001   10000000000000000000000    0x40C00000  1086324736
12.0    0  10000010   10000000000000000000000    0x41400000  1094713344
```

Same fraction, increasing exponent. Each step doubles the value. The bit patterns increase in the same order as the numeric values.

Because the exponent comes before the fraction in the bit layout, a larger exponent always wins—exactly what we want. This was a deliberate design choice: positive floats can be compared using integer comparison hardware.^[This clever design is attributed to William Kahan, the primary architect of IEEE 754.]

##### The Problem with Negative Floats

For negative floats, the ordering breaks. IEEE 754 uses sign-magnitude representation: the sign bit is separate, and the remaining bits encode the magnitude the same way as for positives.

**Same exponent, increasing fraction (negative):**
```
Value   S  Exponent   Fraction                   Hex        Unsigned
-1.0    1  01111111   00000000000000000000000    0xBF800000  3212836864
-1.25   1  01111111   01000000000000000000000    0xBFA00000  3214934016
-1.5    1  01111111   10000000000000000000000    0xBFC00000  3217031168
-1.75   1  01111111   11000000000000000000000    0xBFE00000  3219128320
```

Numerically: $-1.0 > -1.25 > -1.5 > -1.75$

But the bit patterns: $3212836864 < 3214934016 < 3217031168 < 3219128320$

The ordering is *reversed*. When the fraction increases, the magnitude increases—but for negative numbers, larger magnitude means *smaller* value.

**Same fraction, increasing exponent (negative):**
```
Value   S  Exponent   Fraction                   Hex        Unsigned
-1.5    1  01111111   10000000000000000000000    0xBFC00000  3217031168
-3.0    1  10000000   10000000000000000000000    0xC0400000  3225419776
-6.0    1  10000001   10000000000000000000000    0xC0C00000  3233808384
-12.0   1  10000010   10000000000000000000000    0xC1400000  3242196992
```

Numerically: $-1.5 > -3.0 > -6.0 > -12.0$

Bit patterns: $3217031168 < 3225419776 < 3233808384 < 3242196992$

Same problem. The core issue: for negative floats, "larger bit pattern = larger magnitude = more negative = smaller number."

So here's what we need:
1. All positive floats should sort higher than all negative floats
2. Among positive floats, bit-pattern order is already correct
3. Among negative floats, we need to *reverse* the bit-pattern order

The solution: **flip the sign bit for positive floats, flip ALL bits for negative floats**. Flipping the sign bit on positives makes them sort after negatives (since 1 > 0). Flipping all bits for negatives reverses their order, which is exactly what we need since their magnitude ordering is backwards.

```c
__device__ __host__ unsigned int float_to_sortable(float f) {
    unsigned int bits = __float_as_uint(f);
    // If sign bit is set (negative), flip all bits
    // If sign bit is clear (positive), flip only sign bit
    unsigned int mask = (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return bits ^ mask;
}

__device__ __host__ float sortable_to_float(unsigned int bits) {
    // Reverse the transformation
    unsigned int mask = (bits & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu;
    return __uint_as_float(bits ^ mask);
}
```

Let's trace through some values:
```
Original float → Bits                                → Transformed
         -3.0 → 1 10000000 10000000000000000000000 → 0 01111111 01111111111111111111111
         -2.0 → 1 10000000 00000000000000000000000 → 0 01111111 11111111111111111111111
         -1.0 → 1 01111111 00000000000000000000000 → 0 10000000 11111111111111111111111
          0.0 → 0 00000000 00000000000000000000000 → 1 00000000 00000000000000000000000
          1.0 → 0 01111111 00000000000000000000000 → 1 01111111 00000000000000000000000
          2.0 → 0 10000000 00000000000000000000000 → 1 10000000 00000000000000000000000
          3.0 → 0 10000000 10000000000000000000000 → 1 10000000 10000000000000000000000
```

The transformed values are now in the correct order for unsigned comparison. Negative floats end up in the lower half (their sign bit became 0), positive floats in the upper half (their sign bit became 1), and within each group the ordering is correct.

