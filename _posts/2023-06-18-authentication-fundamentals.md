---
title: How does the internet work?
date: 2023-06-07 08:42:00 +0530
categories: [Learning, Cloud]
tags: [Essentials]
math: false
pin: false
image:
  path: https://wallpaperaccess.com/full/1445467.jpg
  lqip: data:image/webp;base64,UklGRpoAAABXRUJQVlA4WAoAAAAQAAAADwAABwAAQUxQSDIAAAARL0AmbZurmr57yyIiqE8oiG0bejIYEQTgqiDA9vqnsUSI6H+oAERp2HZ65qP/VIAWAFZQOCBCAAAA8AEAnQEqEAAIAAVAfCWkAALp8sF8rgRgAP7o9FDvMCkMde9PK7euH5M1m6VWoDXf2FkP3BqV0ZYbO6NA/VFIAAAA
  alt: How does the internet work?
---

# Reference

[Cloudfare Learning](https://www.cloudflare.com/en-gb/learning/network-layer/what-is-the-network-layer/)

# What is a network?

In the simplest of terms, a network is nothing but a bunch of connect devices that can communicate with one another. Usually all these devices are connected to a central hub - for example a **router**.

A network can be of varied sizes and consequently be organized in different ways. Large networks tend to have subnetworks like how an ISP manages thousands of IP addresses and connected devices.

# What is the internet?

> The Internet is a vast, sprawling collection of networks that connect to each other. In fact, the word "Internet" could be said to come from this concept: _inter_connected _net_works.

Since computers connect to each other within networks and these networks also all connect with each other, one computer can talk to another computer in a faraway network thanks to the Internet. This makes it possible to rapidly exchange information between computers across the world.

# What happens at the network layer?

> The layer at which any communication between connected devices in network happen is called the network layer.

For example when Alice send a message on an IM to Bob over a network, there message is first broken down into smaller pieces called "packets" and sent over the network. These packets are then reassembled in Bob's device.

## Packets

The question probably in our minds is, how does Bob's device know how to put the packets together?

At the network layer, networking software attaches a header to each packet when the packet is sent out over the Internet, and on the other end, networking software can use the header to understand how to handle the packet.

A header contains information about the content, source, and destination of each packet (somewhat like stamping an envelope with a destination and return address). For example, an IP header contains the destination [IP address](https://www.cloudflare.com/learning/dns/glossary/what-is-my-ip-address/) of each packet, the total size of the packet, an indication of whether or not the packet has been fragmented (broken up into still smaller pieces) in transit, and a count of how many networks the packet has traveled through.

### Packet Switching

Packets are sent across the Internet using a technique called packet switching. Intermediary routers and switches are able to process packets independently from each other, without accounting for their source or destination. This is by design so that no single connection dominates the network. If data was sent between computers all at once with no packet switching, a connection between two computers could occupy multiple cables, routers, and switches for minutes at a time. Essentially, only two people would be able to use the Internet at a time — instead of an almost unlimited number of people, as is the case in reality.

> Very good video on packets: https://www.youtube.com/watch?v=k8rJFgeuZRw

## Protocols

Connecting two computers, both of which may use different hardware and run different software, is one of the main challenges that the creators of the Internet had to solve. It requires the use of communications techniques that are understandable by all connected computers, just as two people who grew up in different parts of the world may need to speak a common language to understand each other.

This problem is solved with standardized protocols. In networking, a protocol is a standardized way of doing certain actions and formatting data so that two or more devices are able to communicate with and understand each other.

There are protocols for:
- Sending packets between devices on the same network (Ethernet)
- Sending packets from network to network ([IP](https://www.cloudflare.com/learning/ddos/glossary/internet-protocol/))
- Ensuring those packets successfully arrive in order ([TCP](https://www.cloudflare.com/learning/ddos/glossary/tcp-ip/))
- Formatting data for websites and applications ([HTTP](https://www.cloudflare.com/learning/ddos/glossary/hypertext-transfer-protocol-http/))

In addition to these foundational protocols, there are also protocols for routing, testing, and [encryption](https://www.cloudflare.com/learning/ssl/what-is-encryption/). And there are alternatives to the protocols listed above for different types of content — for instance, streaming video often uses [UDP](https://www.cloudflare.com/learning/ddos/glossary/user-datagram-protocol-udp/) instead of TCP.

Because all Internet-connected computers and other devices can interpret and understand these protocols, the Internet works no matter who or what connects to it.

# What physical infrastructure makes the Internet work?

A lot of different kinds of hardware and infrastructure go into making the Internet work for everyone. Some of the most important types include the following:

-   [Routers](https://www.cloudflare.com/learning/network-layer/what-is-a-router/) forward packets to different computer networks based on their destination. Routers are like the traffic cops of the Internet, making sure that Internet traffic goes to the right networks.
-   [Switches](https://www.cloudflare.com/learning/network-layer/what-is-a-network-switch/) connect devices that share a single network. They use packet switching to forward packets to the correct devices. They also receive outbound packets from those devices and pass them along to the right destination.
-   Web servers are specialized high-powered computers that store and serve content (webpages, images, videos) to users, in addition to hosting applications and databases. Servers also respond to [DNS](https://www.cloudflare.com/learning/dns/what-is-dns/) queries and perform other important tasks to keep the Internet up and running. Most servers are kept in large data centers, which are located throughout the world.

# Standardization of network communication

Network communication can be summarized and encapsulated using a couple of models which are quite prevalent in the community.

## OSI Model

The Open Systems Interconnection (OSI) Model is a description of how the Internet works. It breaks down the functions involved in sending data over the Internet into seven layers. Each layer has some function that prepares the data to be sent over wires, cables, and radio waves as a series of bits.

The seven layers of the OSI model are:

-   **7. Application layer:** Data generated by and usable by software applications. The main protocol used at this layer is [HTTP](https://www.cloudflare.com/learning/ddos/glossary/hypertext-transfer-protocol-http/).
-   **6. Presentation layer:** Data is translated into a form the application can accept. Some authorities consider [HTTPS encryption](https://www.cloudflare.com/learning/ssl/what-is-https/) and decryption to take place at this layer.
-   **5. Session layer:** Controls connections between computers (this can also be handled at layer 4 by the [TCP protocol](https://www.cloudflare.com/learning/ddos/glossary/tcp-ip/)).
-   **4. Transport layer:** Provides the means for transmitting data between the two connected parties, as well as controlling the quality of service. The main protocols used here are TCP and [UDP](https://www.cloudflare.com/learning/ddos/glossary/user-datagram-protocol-udp/).
-   **3. Network layer:** Handles the routing and sending of data between different networks. The most important protocols at this layer are IP and ICMP.
-   **2. Data link layer:** Handles communications between devices on the same network. If layer 3 is like the address on a piece of mail, then layer 2 is like indicating the office number or apartment number at that address. Ethernet is the protocol most used here.
-   **1. Physical layer:** Packets are converted into electrical, radio, or optical pulses and transmitted as bits (the smallest possible units of information) over wires, radio waves, or cables.

## TCP/IP Model

In the TCP/IP model, the four layers are:

-   **4. Application layer:** This corresponds, approximately, to layer 7 in the OSI model.
-   **3. Transport layer:** Corresponds to layer 4 in the OSI model.
-   **2. Internet layer:** Corresponds to layer 3 in the OSI model.
-   **1. Network access layer:** Combines the processes of layers 1 and 2 in the OSI model.

But where are OSI layers 5 and 6 in the TCP/IP model? Some sources hold that the processes at OSI layers 5 and 6 either are no longer necessary in the modern Internet, or actually belong to [layers 7](https://www.cloudflare.com/learning/ddos/what-is-layer-7/) and 4 (represented by layers 4 and 3 in the TCP/IP model).

