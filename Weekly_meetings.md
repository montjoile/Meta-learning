# Weekly meetings

### Credit
This template is partially derived from "Whitaker Lab Project Management" by Dr. Kirstie Whitaker and the Whitaker Lab team, used under CC BY 4.0. 

------


## Week 5
**Date: 26-06-2019**

* **What did you achieve?**
  Read literature review about dataset similarity for feature selection. I found some papers for automatic feature selection methods, but I decided to first try with statistical meta-features, like std, # of classes, image size, etc.
 

* **What did you struggle with?**
  My pc ran out of space in the hard disk due to the volume of the datasets and was crashing constantly, so I had to replicate my environment in the HPC. 


* **What would you like to work on next week?**
  Finish the visualization functions and double check the final results. I am expecting to end up with at leat 10 meta-features that are very similar between datasets. I want to perform various measurements using different subsets of each dataset to see if that changes the results. The question I want to answer with this is if the variability in the datasets further affects the performance of the meta-learner, but that would be tested at later stage. For now, I just want to code the functions to use them later. 


* **Where do you need help from Ali?**
  Should I consider euclidean distance as a good metric for similarity? 


* **Where do you need help from Veronika?**
  Some images from the datasets have annotations, as colored regions to identify the class of the image. I wonder if this might be 'cheating' if I choose the colors of the images as a meta-feature.
  
 