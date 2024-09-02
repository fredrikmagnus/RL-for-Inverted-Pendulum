NOTE:
Add corrected reset conditions based on new parameters. 
Make model act methods return force. 
Make new step function which takes force, not action. 
Refactor input parameters for each model.
Make a better reward function for DDPG.

Refactor the method of passing class parameters. Do some research for the best way to do it. We could:
    a) Refactor the input parameters for each model so that each model has its own section. 
    b) Directly pass these parameters to the model class.