---
title: Syntax highlighting & Mathematical markup in HTML
date: 2020-08-27 16:45:00 +0530
categories: [Tutorial]
tags: [HTML]
math: true
pin: false
image:
  path: https://images.unsplash.com/photo-1461749280684-dccba630e2f6?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8Y29kaW5nfGVufDB8fDB8fHww&w=1000&q=80
  lqip: data:image/webp;base64,UklGRpoAAABXRUJQVlA4WAoAAAAQAAAADwAABwAAQUxQSDIAAAARL0AmbZurmr57yyIiqE8oiG0bejIYEQTgqiDA9vqnsUSI6H+oAERp2HZ65qP/VIAWAFZQOCBCAAAA8AEAnQEqEAAIAAVAfCWkAALp8sF8rgRgAP7o9FDvMCkMde9PK7euH5M1m6VWoDXf2FkP3BqV0ZYbO6NA/VFIAAAA
  alt: Syntax highlighting & Mathematical markup in HTML
---

## Why do I need syntax highlighting?

Syntax highlighting of "code" blocks in a HTML is useful if say you want to create a blog on a programming language and you would like to convey to the readers the code clearly.

Syntax highlighting in HTML is super simple and improves the overall readability of the page content.

## HighlightJS

Though there are a number of solutions available, in my experience the simplest one which gets the job done is highlightJS.

[HighlightJS](https://highlightjs.org/) is a simple syntax highlighting javascript library.

To add syntax highlighting on your webpage, add this code within the head tag.

    <link rel="stylesheet" href="//cdnjs.cloudflare.com/ajax/libs/highlight.js/9.9.0/styles/androidstudio.min.css">
    <script src="//cdnjs.cloudflare.com/ajax/libs/highlight.js/9.9.0/highlight.min.js"></script>
    <script>hljs.initHighlightingOnLoad();</script>
    

First we load up the CSS & JS script for highlightJS, next we initialize them so they can be used.

To add syntax highlighting to the following python code snippet, the html code will be:

    <pre><code class="python hljs">def map(key, value):
      # key: document name
      # value: document contents
      for each word w in value:
        EmitIntermediate(w, "1");
    
    def reduce( key, values):
      # key: a word
      # values: a list of counts
      int result = 0;
      for each v in values:
        result += ParseInt(v);
        Emit(AsString(result));</code></pre>
    

The result would look something like this:

    def map(key, value):
      # key: document name
      # value: document contents
      for each word w in value:
        EmitIntermediate(w, "1");
    
    def reduce( key, values):
      # key: a word
      # values: a list of counts
      int result = 0;
      for each v in values:
        result += ParseInt(v);
        Emit(AsString(result));
    

For more information on syntax highlighting for other languages, you can check the [highlightJS demo](https://highlightjs.org/static/demo/).

## Why do I need mathematical markup?

Although the mark-up language HTML has a large repertoire of tags, it does not cater for math. With no means of using HTML tags to mark up mathematical expressions, authors have resorted to drastic means. For example, a popular method involves inserting images - literally snap shots of equations taken from other packages and saved in GIF format - into technical documents which have a mathematical or scientific content.

## Mathematical Markup Langauge (MathML)

[MathML](https://www.w3.org/Math/whatIsMathML.html) is intended to facilitate the use and re-use of mathematical and scientific content on the Web. MathML can be used to encode both the presentation of mathematical notation for high-quality visual display, and mathematical content.

To add MathML to your webpage, we need to import the libraries for the same within the head tag. Both CSS and the JS script can be imported with a single link.

    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML-full"></script>
    

MathML needs to be initialized and this can be done by adding this script within the head tag.

    <script type="text/x-mathjax-config">
      //Note the above <script> attribute type="text/x-mathjax-config" 
      MathJax.Hub.Config({
          tex2jax: {inlineMath: [["$","$"],["\\(","\\)"]]}
      });
    </script>
    

For example to display the following mathematical equation:

$$(X^2 + Y^2) \le \Delta$$

The HTML code for the same would be:

    <p>$$(X^2 + Y^2) \le \Delta$$</p>
    

Click on the hyperlink to find more information on the [LaTeX commands for the math symbols](https://oeis.org/wiki/List_of_LaTeX_mathematical_symbols).
