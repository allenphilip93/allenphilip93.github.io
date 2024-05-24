---
title: Man In The Middle (MITM) attack
date: 2023-06-26 11:22:00 +0530
categories: [Learning, Cloud]
tags: [Essentials]
math: false
pin: false
image:
  path: https://s38063.pcdn.co/wp-content/uploads/2023/10/BlogPost-MPM-ManInTheMiddle-LC-85410.jpg
  lqip: data:image/webp;base64,UklGRpoAAABXRUJQVlA4WAoAAAAQAAAADwAABwAAQUxQSDIAAAARL0AmbZurmr57yyIiqE8oiG0bejIYEQTgqiDA9vqnsUSI6H+oAERp2HZ65qP/VIAWAFZQOCBCAAAA8AEAnQEqEAAIAAVAfCWkAALp8sF8rgRgAP7o9FDvMCkMde9PK7euH5M1m6VWoDXf2FkP3BqV0ZYbO6NA/VFIAAAA
  alt: Man In The Middle (MITM) attack
---

## What is MITM?

A man in the middle (MITM) attack is a general term for when a perpetrator positions himself in a conversation between a user and an application—either to eavesdrop or to impersonate one of the parties, making it appear as if a normal exchange of information is underway.

The goal of an attack is to steal personal information, such as login credentials, account details and credit card numbers. Targets are typically the users of financial applications, SaaS businesses, e-commerce sites and other websites where logging in is required.

## Example of MITM attack

Let's consider an example of a man-in-the-middle (MITM) attack on an unsecure HTTP connection:

1.  **Initial Connection**:
    -   Alice wants to access a website (e.g., example.com) over an unsecure HTTP connection.
    -   The website is hosted by Bob.
    
2.  **DNS Spoofing**:
    -   Eve, the attacker, intercepts the DNS resolution process. When Alice tries to resolve the domain name example.com, Eve responds with a fake IP address that belongs to her.

3.  **Request Forwarding**:   
    -   Alice's browser, believing it has reached the legitimate website, sends an HTTP request to the IP address provided by Eve.
    -   The request contains information like the desired resource (e.g., example.com/home.html) and any user input.

4.  **Intercepting the Request**:
    -   Eve, sitting between Alice and Bob, intercepts the HTTP request sent by Alice before it reaches the actual web server.

5.  **Creating a New Request**:
    -   Eve creates a new HTTP request to Bob's web server, mimicking the request made by Alice.
    -   Eve acts as a client to Bob's server and establishes a separate connection.

6.  **Server Response**:
    -   Bob's server responds to Eve's request with the appropriate HTTP response, containing the requested resource (e.g., home.html).

7.  **Response Forwarding**:
    -   Eve intercepts the response from Bob's server and creates a new HTTP response to send back to Alice.
    -   She can inject malicious code, modify the content, or even replace the entire response with a different one.

## Intercepting the DNS resolution process

There are several methods an attacker can employ to intercept the DNS resolution process. Here are a few common techniques used in man-in-the-middle (MITM) attacks:

1.  **DNS Cache Poisoning**: In this method, the attacker exploits vulnerabilities in DNS servers or the DNS caching mechanism to inject fake DNS records into the cache. When a user's device queries the DNS server for a specific domain, it retrieves the malicious DNS record instead of the legitimate one, directing the user to the attacker-controlled IP address.
    
2.  **ARP Spoofing/ARP Poisoning**: Address Resolution Protocol (ARP) spoofing involves sending fake Address Resolution Protocol messages to network devices, tricking them into associating the attacker's MAC address with the IP address of the legitimate DNS server. Consequently, all DNS traffic from the victim's device is routed through the attacker, allowing them to manipulate the DNS responses.
    
3.  **DNS Hijacking**: In this technique, the attacker compromises a DNS server and modifies its settings to redirect DNS queries to their own rogue DNS server. When the victim's device sends a DNS query, it reaches the compromised DNS server, which responds with malicious or incorrect IP addresses, leading the victim to unintended destinations.
    
4.  **Rogue Wi-Fi Access Points**: Attackers can set up rogue Wi-Fi access points with deceptive names that closely resemble legitimate networks, such as public hotspots or the victim's trusted network. When users connect to these rogue networks, the attacker gains control over their network traffic, including DNS queries, enabling them to intercept and manipulate the DNS resolution process.
    
5.  **Malware or Browser Exploits**: Attackers can infect a victim's device with malware or exploit vulnerabilities in their web browser. This allows them to modify the DNS settings locally on the compromised device, redirecting DNS queries to their preferred IP address or DNS server.
