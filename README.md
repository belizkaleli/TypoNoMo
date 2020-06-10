# TypoNoMo
A chrome extension that uses machine learning and more to check your tweet before you post it to identify typo URLs and warn you if needed.

## Typo URLs
Twitter detects links in your tweet text before you post them and highlights them in color blue. 
Twitter's only mechanism to identify URLs is to check the [Top Level Domain](https://developer.mozilla.org/en-US/docs/Glossary/TLD) in your link. 
There are thousands of Top Level Domains ([Here](https://data.iana.org/TLD/tlds-alpha-by-domain.txt) is a list.) which are actual English words so it is pretty easy for a user to accidentally type a URL by making a typo.

For example following is a tweet that contains a typo URL:  
__Tweet__: "Good for you.you're not blind and you can see!"  
__Typo URL__: "you.you"  

## Paper
Typo URL phenomena poses many security threats. This issue is explained in detail in our paper. (put link)

## What does TypoNoMo do?
This chrome extension detects possible typo URLs in your tweet text before you post it and warns you by pointing out the possible typo URL. You can go ahead and click "Continue" to ignore the warning or you can click "Cancel" to take one more look at your tweet before posting.
The extension does not bother you if there are no possible typo URLs in your tweet.

*A Boston University [SeclaBU](https://seclab.bu.edu/) project.*
