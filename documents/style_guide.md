
# Style Guide 

We are following to the best of our abilities the [PEP8](https://www.python.org/dev/peps/pep-0008/) and [numpy docstring](https://numpydoc.readthedocs.io/en/latest/format.html) style convention. Note: Pycharm has an inbuild checking framework that will help you follow the style guide. 

The General convention is as follows:

> Class variable: NameOfClass

> Function/Methods/Variable: 
        Private: _name_private_function
        Public:  name_public_function

> File name: name_file

> Agent/Experiments: author_date

> Examples: name_examples

In what follows we describe the general format for each of the three main components of NeuralPlayground. We begin with the Agents component, then proceed to describing Arenas and then finally Experiments.

## Agents

To add an agent to the library begin by creating a file with the naming convention of "author_date.py" where "author" is the name of the lead author who from the work which introduced the agent/model and "date" is the year the work was published. In this file implement the new class for the agent with class naming format "AuthorDate". Ensure that this class inherits the "AgentCore" class found in "agent_core.py". Consequently your new class will inherit the minimal set of attributes and methods necessary to function flexibly within the other pipelines implemented by NeuralPlayground. These core attributes are as follows:

> * `model_name` : *str* 
>     - The name of the new agent class you are implementing. Can be any valid string and will usually be used for labelling plots or printing to terminal.
> * `mod_kwargs`: *dict*
>     - Dictionary of keyword arguments passed to the `__init__()` function during instantiation of the object.
> * `metadata`: *dict*
>     - Dictionary reserved for containing specific description details for each model. By default it just captures all keyword arguments passed in during 	    instantiation of a new object.
> * `obs_history`: *list* 
>     - List of the agent's past observations obtained while interacting with the environment. This is populated progressively with each call of the act method.
> * `global_steps`: *int*
>     - Records the number of updates done to the weights or parameters of a model implemented within the class *if* one is used.

Additionally the class will also inherit the necessary methods that the rest of the library will use to interface with its objects. These are as follows:

> * `__init__( )`
>     - Accepts:  
>         - `model_name` : *str* 
>             - Default: "default_model" 
>         - `**mod_kwargs`: *dict* 
>             - Default: {}
>     - Returns: None
>     - Description: Function which initialises an object of the class. Naming the object is the only required input. All other inputs are passed as keyword 	arguments that are used to create metadata or custom attributes or provide further functionality to custom methods.
>
> * `reset()` <!-- in the code the act function populates obs_history but this doesn't reset it -->
>     - Accepts: None
>     - Returns: None
>     - Description: Erases all memory from the model, re-initialises all relevant parameters and builds the original object from scratch.
>
> * `neural_response()`
>     - Accepts: None
>     - Returns: None
>     - Description: Returns the neural representation of the model performing the given task. Output will be compared against real experimental data.
>
> * `act()`
>     - Accepts:
>         - `obs`: *np.array()*
>             - Default: None 
>             - Description: Observation from the environment class needed to choose the right action
>         - `policy_func`: *function* 
>             - Default: None
>             - Description: Arbitrary function that represents a custom policy that receives and observation and gives an action
>     - Returns:
>         - `action`: *np.array(dtype=float)*
>         - Description: The action value indicating the direction the agent moves in 2d space (np array will have shape of (2,))
>     - Description: Chooses and executes and action for the agent. Typically depends on the agent's observations of the environment. 
>
> * `update()`
>     - Accepts: None
>     - Returns: None
>     - Description: Alters the parameters of the model (if there are any) likely based on the observation history to simulate learning.
>
> * `save_agent()`
>     - Accepts:
>         - `save_path`: *str*
>             - Default: None 
>             - Description: Path to the file where the objects state and information will be saved
>     - Returns: None
>     - Description: Saves the current state and object information to be able to re-instantiate the environment from scratch. 
>
> * `restore_agent()`
>     - Accepts: 
>         - `save_path`: *str* <!-- bad variable name --> 
>             - Default: None  
>             - Description: Path to the file where the objects state and information will be restored from. 
>     - Returns: None 
>     - Description: Restores and re-instantiate the environment from scratch using the state and object information stored in the file at `save_path`. 
>
> * `__eq__()` <!-- check what this does -->
>     - Accepts:
>         - `other`: *dict*
>             - Default: None
>             - Description: <!-- todo -->
>     - Returns: *bool*
>         - Description: True if dictionaries are the same, False if they are different.
>     - Description: Determines whether two dictionaries are the same or equal.

## Environment/Arena

To add an environment we follow a similar convention to the Agent class. If the environment is based on a publication begin by creating a file with the naming convention of "author_date.py" where "author" is the name of the lead author and "date" is the year the work was published. Otherwise create a file with a descriptive name such as "connected_rooms.py". There are two possible classes which a new class could inherit to obtain the minimal set of attributes and methods necessary to function flexibly within the other pipelines implemented by NeuralPlayground. Firstly "agent_core.py" which provides the most basic interface necessary. Secondly, "simple2d.py" provides a richer interface which can be used to create 2-dimensional navigation based domains and inherits "agent_core.py". We will begin by describing "agent_core.py" and then move on to describe "simple2d.py".

### agent_core.py
The core attributes are as follows:

> * `state` : *array* <!-- all these vars say "Define within each subclass for specific environments" when used in functions which kind of defeats the point -->
>     - Empty array for this abstract class. Designed to contain the present state of the environment.
> * `history`: *list*
>     - Contains the transition history of all states in the environment. Differs from the history of an agent which may not fully observe the full state. Here this is the history of the full state of the environment.
> * `time_step_size`: *float*
>     - The number of seconds by which the "in-world" time progresses when calling the `step()` method.
> * `metadata`: *dict* 
>     - Contains extra metadata that might be available in other classes.
> * `env_kwags`: *dict*
>     - Arguments given to the init method.
> * `global_steps` : *int* 
>     - Counts the number of calls to the `step()` method. Set to 0 when calling `reset()`.
> * `global_time`: *float*
>     - Total "in-world" time simulated through calls to the `step()` method since the last reset. Then `global_time = time_step_size * global_steps`.
> * `observation_space`: *gym.spaces*
>     - Specifies the range of observations which the environment can generate as in OpenAI Gym.
> * `action_space`: *gym.spaces* 
>     - Specifies the range of valid actions which an agent can take in the environment as in OpenAI Gym.

Additionally the class will also inherit the necessary methods that the rest of the library will use to interface with its objects. These are as follows:

> * `__init__( )`
>     - Accepts:  
>         - `environment_name` : *str* 
>             - Default: "Environment"
>         - `time_step_size` : *float*
>             - Default: 1.0
>         - `env_kwargs`: *dict* 
>             - Default: {}
>     - Returns: None
>     - Description: Function which initialises an object of the class. Naming the object is the only required input. All other inputs are passed as keyword arguments that are used to create metadata or custom attributes or provide further functionality to custom methods.
>
> * `make_observation()`
>     - Accepts: None
>     - Returns:
>         - `self.state` : *array*
>             - Description:  Variable containing the state of the environment (eg. position in the environment)
>     - Description: Takes the state and returns an array of sensory information for an agent. In more complex cases, the observation might be different from the internal state of the environment or partially observable.
>
> * `step()`
>     - Accepts:
>         - `action` : *array*
>             - Description: Type is currently set to match environment but any type can be used as long as the function is able to still return the necessary variables.  
>     - Returns:
>         - `observation`: *array*
>             - Description: Any set of observation that can be encountered by the agent in the environment (position, visual features,...)  
>         - `self.state` : *array*
>             - Description:  Variable containing the state of the environment (eg. position in the environment)
>         - `reward`: int <!-- why int? -->
>     - Description: Runs the environment dynamics resulting from a given action. Increments global counters and returns the resultant observation by the agent, new state of the environment (which is not necessarily the same as the agent's observation of the environment) and the reward.
> 
> * `_increase_global_step()`
>     - Accepts: None
>     - Returns: None
>     - Description: Increments the `self.global_steps` and `self.global_time` counters.
> 
> * `reset()`
>     - Accepts: None
>     - Returns:
>         - `observation`: *array*
>             - Description: Any set of observation that can be encountered by the agent in the environment (position, visual features,...)
>         - `self.state` : *array*
>             - Description:  Variable containing the state of the environment (eg. position in the environment) 
>     - Description: Re-initialize state. Returns observation and state after the reset. Also returns time and step counters to 0.
> 
> * `save_environment()`
>     - Accepts:
>         - `save_path`: *str*
>             - Description: Path to the file that the environment state will be saved to 
>     - Returns: None
>     - Description: Save current variables of the environment to re-instantiate it in the same state later
>
> * `restore_environment()`
>     - Accepts:
>         - `save_path`: *str* <!-- var name -->
>             - Description: Path to the file that the environment state can be retrieved from 
>     - Returns: None
>     - Description: Restores the variables of the environment based on the stated save in `save_path`
>
> * `__eq__()`
>     - Accepts:
>         - `other`: *Environment*
>             - Description: Another instantiation of the environment
>     - Returns: *bool*
>     - Description: Checks if two environments are equal by comparing all of its attributes. True if self and other are the same exact environment
>
> * `get_trajectory_data()`
>     - Accepts: None 
>     - Returns:
>         - `self.history`: *list*
>             - Description: Contains the transition history of all states in the environment. Differs from the history of an agent which may not
>     - Description: Returns state history of the environment since last reset.
>
> * `reward_function()`
>     - Accepts:
>         - `action`: *array*
>             - Description: Type is currently set to match environment but any type can be used as long as the function is able to still return the necessary variables. Some encoding of a valid action in the environment which can determine the value of the action in the given state.
>         - `state`: *array*
>             - Description: Variable containing a possible state of the environment (eg. position in the environment). Does not have to be the current state of the environment.
>     - Returns:
>         - `reward`: *float*
>             - Description: Reward of taking the given action in the given state.
