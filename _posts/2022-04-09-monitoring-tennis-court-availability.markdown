---
layout: post
title: "Monitoring tennis court availability"
date: 2022-04-09 13:39:24 +0100
categories: jekyll update
---

### Motivation

I've been introduced to tennis recently and I love it. I'm struggling however with booking tennis courts. Most of the tennis courts in Paris are owned by the city of Paris and booking one is a pain in the ass because the demand is very high.

Since everyone is working during the week, it's almost impossible to find a time slot in the evening after work or during weekends. The only way to reserve such a time slot is to book it before everyone else.

Every morning at exactly 8:00 AM, the time slots of 1 day that is 1 week from now are made open for reservation. At 8:01 AM most time slots for that day are already gone.

I was tired of having to wake up early to do this, so I wanted to benefit from my coding skills to solve this problem.

At first I thought about creating a bot that automatically does the booking for me, but I quickly understood that too much effort was needed to automate that (including automating the payment)

People cancel their reservations quite often, so interesting time slots are made available at unpredictable times. If I can be notified whenever an interesting time slot becomes available, that would allow me to reserve a tennis court without having to compete against an army of tennis players who have woken up 8:00 AM to beat me to booking the best time slots.

### Proof of concept

Is it even possible to monitor the available time slots? Is there a way to scrape that information off the website of Paris tennis courts automatically?

It turns out the answer yes. Check out [this nice page here](https://tennis.paris.fr/tennis/jsp/site/Portal.jsp?page=recherche&view=planning&name_tennis=Elisabeth). It shows the available courts in a specific sports facility (Elisabeth) in nicely formatted HTML table.

I can change the name of the sports facility in the URL and that gives me the available courts for another facility. I'm particularly interested in two sports facilities that are close to me (Elisabeth and Poterne des Peupliers).

If we visit that page from a chrome browser and inspect network activity when changing the selected date, we can intercept the request being sent from the browser. After removing unnecessary headers to simplify the request, here's what we're left with:

```shell
curl --location --request POST 'https://tennis.paris.fr/tennis/jsp/site/Portal.jsp?page=recherche&action=ajax_load_planning' \
--header 'Content-Type: application/x-www-form-urlencoded; charset=UTF-8' \
--data-raw 'date_selected=08%2F04%2F2022&name_tennis=Elisabeth'
```

Note that there are two parameters that are passed through the request: the selected date and the name of the sports facility.

### Parsing the HTML

I used the BeautifulSoup Python package to parse the HTML of the page. The parsing method takes as input a date and the name of a sports facility and returns a list of available time slots. A time slot is defined as a start time, end time, facility name, court number, and date.

```python
@dataclass
class TimeSlot:
    start: int
    end: int
    facility: str
    court: int
    date: datetime
```

### Fetching available courts that satisfy my pereferences

Since I can fetch the available time slots at a given facility on a given day (up to one week ahead), I can now automate scanning availabilities for the whole week in sports facilities that are close to where I live and filter to keep only the good slots.

I can setup preferences such as minimum start time and select specific courts, etc. At the time of writing, my preferences are:

* Only Elisabeth and Poterne des Peupliers sports facilities (closest to where I live)
* If during the week, only time slots after 8 PM
* If during the weekend, only time slots after 12 PM

### Making it run on its own

Now that I can automatically check for new availabilities, I need a way to run the script for checking available time slots automatically and send me an email whenever new time slots are available.

An easy way to implement this for free is to use AWS Services. The services I need are:

* Lambda functions: allows to execute the script that checks availabilities and sends an email without having to rent a server.
* CloudWatch Events: triggers the Lambda function periodically at a configurable time interval.
* Simple Email Service: used to send the email.

### Avoiding spam

One small problem I had to deal with is how to avoid sending the same email over and over. I would like my script to check for availability as often as possible in order for me to be able to book a slot quickly. However, if it sends me an email every time it runs, my inbox will be drowned with emails.

Therefore, I need a way to store state in the Lambda function. The state would be simply the last email, and then whenever a the script runs, it checks whether any of the current availabilities are not in the last email, and only then it sends an email.

It turns out there is a neat trick that allows to store state within the Lambda function without using any storage services. That is the `UpdateFunctionCode` functionality in Lambda. The idea is that you can make the Lambda function update its own code while it's running.

Here's how to use that to avoid sending the same email over and over:

1. When the Lambda function runs, it tries to open a file `last_email.txt`. If the file is not there, then assume that this is the first run.
2. Fetch the interesting availabilities and format them into an email.
3. If the new email contains availabilities that were not in the last email, send the new email.
4. Save the new email into `last_email.txt` and update the Lambda function code to reflect that change.


### Result

The result is a bot that is completely free, runs on AWS once every minute, and sends me an email whenever someone cancels a reservation that satisfies my criteria for a time slot. Here's an example email:

```
Hurray! Tennis courts are available!

13/04/2022 21h-22h Elisabeth Court 5
16/04/2022 21h-22h Elisabeth Court 9
16/04/2022 20h-21h Poterne des Peupliers Court 1
16/04/2022 20h-21h Poterne des Peupliers Court 2
16/04/2022 21h-22h Poterne des Peupliers Court 1
16/04/2022 21h-22h Poterne des Peupliers Court 2
17/04/2022 12h-13h Elisabeth Court 1
17/04/2022 12h-13h Elisabeth Court 2
17/04/2022 12h-13h Elisabeth Court 6
17/04/2022 12h-13h Elisabeth Court 7
17/04/2022 12h-13h Poterne des Peupliers Court 2

Link to reserve: https://tennis.paris.fr/tennis/jsp/site/Portal.jsp?page=recherche&action=rechercher_creneau
```
