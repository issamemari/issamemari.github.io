---
layout: post
title:  "No space left on device during pip install"
date:   2022-03-16 17:04:24 +0100
categories: jekyll update
---

The PyTorch package has been getting larger and larger with newer versions. A common problem I have when installing PyTorch using pip is that I run out of disk space during installation.

```console
imemari@gpu02.ds:~/watches/watch_keypoint_detection$ pip install torch
Could not install packages due to an EnvironmentError: [Errno 28] No space left on device
```

This error seems strange because I'm working on a server with massive amounts of disk storage. Indeed, looking at the amount of available disk space, there are 42 GB available on the `/srv` partition, which is where my home directory is located:

```console
imemari@gpu02.ds:~/watches/watch_keypoint_detection$ df -H
Filesystem                           Size  Used Avail Use% Mounted on
udev                                  50G     0   50G   0% /dev
tmpfs                                 10G  902M  9.1G  10% /run
/dev/md2                              21G   16G  3.6G  82% /
tmpfs                                 50G   58k   50G   1% /dev/shm
tmpfs                                5.3M     0  5.3M   0% /run/lock
tmpfs                                 50G     0   50G   0% /sys/fs/cgroup
/dev/sda1                            535M  5.4M  530M   1% /boot/efi
/dev/md4                             458G  417G   42G  91% /srv
tmpfs                                 10G     0   10G   0% /run/user/1111
tmpfs                                 10G     0   10G   0% /run/user/1118
tmpfs                                 10G     0   10G   0% /run/user/1110
```

Since pip caches the installed packages, let's look at where the cache files are stored:

```console
imemari@gpu02.ds:~/watches/watch_keypoint_detection$ pip cache dir
/srv/imemari/.cache/pip
```

So the cache directory is located on the `/srv` partition, which should have enough space for PyTorch, so what's going on?

It seems that pip creates some temporary files in `/tmp` during installation. If we go to `/tmp` and check which partition it resides on:

```console
imemari@gpu02.ds:~/watches/watch_keypoint_detection$ cd /tmp
imemari@gpu02.ds:/tmp$ df -H .
Filesystem      Size  Used Avail Use% Mounted on
/dev/md2         21G   16G  3.6G  82% /
```

It becomes clear what the problem is: `/tmp` directory is located on partition `/dev/md2`, which has only 3.6G of free space

The solution is to force pip to use a different tmp directory that resides on a partition where we have a lot of free space.

Create a `tmp` directory in the home folder:

```console
imemari@gpu02.ds:/tmp$ cd ~
imemari@gpu02.ds:~$ mkdir tmp
```

We can use the environment variable `TMPDIR` to set the tmp directory used by pip

```console
TMPDIR=~/tmp pip install torch
```

And that solves the problem.
