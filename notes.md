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

REFERENCE README: https://github.com/e-dorigatti/inverted-pendulum/blob/master/README.md

NOTE: I saved working weights for the DDPG model, where it was able to balance the pendulum. The force used was 35.

NOTE: Now trying DDPG with hidden layer sizes of 128 and not 256. Also reducing polyak to 0.9.

TODO: Note that i now made the get state representation part of the environment. This should be updated and removed from the models. 

