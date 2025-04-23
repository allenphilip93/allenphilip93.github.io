---
title: Create a website using NodeJS in 5 mins
date: 2020-08-25 12:35:00 +0530
categories: [Miscellaneous, Tutorial]
tags: [Website, NodeJS]
math: true
pin: false
image:
  path: https://wallpaperaccess.com/full/3909221.jpg
  lqip: data:image/webp;base64,UklGRpoAAABXRUJQVlA4WAoAAAAQAAAADwAABwAAQUxQSDIAAAARL0AmbZurmr57yyIiqE8oiG0bejIYEQTgqiDA9vqnsUSI6H+oAERp2HZ65qP/VIAWAFZQOCBCAAAA8AEAnQEqEAAIAAVAfCWkAALp8sF8rgRgAP7o9FDvMCkMde9PK7euH5M1m6VWoDXf2FkP3BqV0ZYbO6NA/VFIAAAA
  alt: Create a website using NodeJS in 5 mins
---

## What are web applications?

In simple terms a Web Application (or webapp for short) is an application software made available on the web (internet). Any application that is running on a server and can be accessed over the web can be called a webapp.

For example, your company/school website, WhatApp, YouTube etc they're all webapps. In the context of this article we will be talking only about a static website which can be used as a portfolio website or a blog website etc.

## Prerequisites

Ensure that you have NodeJS installed, if not you can find the steps to install the same at [https://nodejs.org/en/download/](https://nodejs.org/en/download/). It would be best to install the 12.x LTS (Long Term Support) version.

NodeJS is a platform built on Chrome's JavaScript runtime for easily building fast and scalable network applications. You can learn more about NodeJS and its wonders at [https://www.freecodecamp.org/news/what-exactly-is-node-js-ae36e97449f5/](https://www.freecodecamp.org/news/what-exactly-is-node-js-ae36e97449f5/)

## Creating the web app

Create a folder and run the following command. Follow the steps to initialize the NodeJS webapp.

    C:\sample>npm init
    This utility will walk you through creating a package.json file.
    It only covers the most common items, and tries to guess sensible defaults.
    
    See `npm help json` for definitive documentation on these fields
    and exactly what they do.
    
    Use `npm install #pkg#` afterwards to install a package and
    save it as a dependency in the package.json file.
    
    Press ^C at any time to quit.
    package name: (sample)
    version: (1.0.0)
    description:
    entry point: (index.js)
    test command:
    git repository:
    keywords:
    author:
    license: (ISC)
    About to write to C:\sample\package.json:
    
    {
      "name": "sample",
      "version": "1.0.0",
      "description": "",
      "main": "index.js",
      "scripts": {
        "test": "echo \"Error: no test specified\" && exit 1"
      },
      "author": "",
      "license": "ISC"
    }
    
    
    Is this OK? (yes) yes

We would like to create a static website so run the following command to create a webapp without any view.

    C:\sample>express --no-view

Now install all the dependencies using the following command.

    C:\sample>npm install

## Running the webapp

To test if the webapp is setup correctly, run the following command.

    C:\sample>npm start
    
    > sample@0.0.0 start C:\sample
    > node ./bin/www

You can access your website using [http://localhost:3000](http://localhost:3000). The webapp can be accessed by anyone in the network using the url http://{machine-name}:3000 or http://{IP-address}:3000.
![blog1a](https://res-3.cloudinary.com/hyfixviip/image/upload/q_auto/v1/ghost-blog-images/blog1a.png)

You can modify the html files and add your own content from the ./public/ folder in your workspace.
