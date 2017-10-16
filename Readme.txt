The project is to create a temperature forecasting system using any machine learning algorithm to get the best results
the project contains multiple files. 
      Data_collecter: this is where the data is collected from the API, http://api.worldweatheronline.com 
                      the file also uses another API for the Post code needed to colect the Data, 
                                            http://api.postcodes.io/postcodes/ (this one is free)
                                            
      Feature_Generator: this is where the features are generated which are the following, 
                         going back (1 -25 hours)
                         means and sums,
                         differences between means for all different possibilities (x,y)
                                     (1,2) (1,3) (1,4) (1,5)...(2,3) (2,4) (2,5) ...(22,23) (23,24)
                                     does not include doubles (1,1) (2,2)
                                     does not include if X > Y
                         after all of this it pickles the data for later use so that the user doesnt have to fetch data after each run
                         ** checking method needs to be updated **
                         
      model: uses the Data generated and splits it based on what day it is. uses 20% for testing and 80% for training.
              ** need to update so it can predict a few hours into the future right now just able to predict what currect day is **
       
      
      
