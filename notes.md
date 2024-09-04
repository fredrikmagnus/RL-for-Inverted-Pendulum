NOTE:
Add corrected reset conditions based on new parameters. 
Make model act methods return force. 
Make new step function which takes force, not action. 
Refactor input parameters for each model.
Make a better reward function for DDPG.

Refactor the method of passing class parameters. Do some research for the best way to do it. We could:
    a) Refactor the input parameters for each model so that each model has its own section. 
    b) Directly pass these parameters to the model class.


Now training the DDPG model. First we started from the down position (DDPG_down weights) and then tried to continue training but to initialize the model from the up position (DDPG_down_up weights). 
Now also added a penalty for high angular velocities.