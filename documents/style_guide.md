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

### arena_core.py
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

### simple2d.py
The overloaded attributes which "simple2d.py" makes more specialized are as follows:

> * `state` : *ndarray*
>     - Contains:
>         - `head_direction` : *ndarray*
>             - Contains the x and y coordinates of the agent's head relative to the body. <!-- check relative to body -->
>         - `position` : *ndarray*
>             - Contains the x and y coordinates of the agent's position in the environment.
>     - Description: Contains the x, y coordinate of the position and head direction of the agent (will be further developed).
> * `history`: *list of dicts*
>     - Saved history over simulation steps (action, state, new_state, reward, global_steps)

In addition some new attributes are added:
> * `room_width`: *int*
>     - Size of the environment in the x axis (0th axis of an ndarray array; the number of rows).
> * `room_depth`: *int*
>     - Size of the environment in the y axis (1st axis of an ndarray array; the number of columns).
> * `observation`: *ndarray*
>     - This is a  fully observable environment. Array of the current observation of the agent in the environment (Could be modified as the environment evolves) `make_observation()` returns the state. <!-- why do you need the var and the function -->
> * `agent_step_size`: *float*
>     - Size of the step when executing movement, `agent_step_size*global_steps` will give a measure of the total distance traversed by the agent.

Additionally your class will also inherit the necessary methods that the rest of the library will use to interface with its objects. "simple2d.py" overloads the following functions from "arena_core.py" These are as follows:

> * `__init__( )` <!-- this function doesn't set defaults for any of the necessary added variables. It also expects them as kwarge, this seems like an issue. It calls the super init on the dictionary but super only sets a value for time_step_size. It also does have an if statement checking these kwarg are there -->
>     - Accepts:  
>         - `environment_name` : *str* 
>             - Default: "2DEnv"
>         - `time_step_size` : *float*
>             - Default: 1.0
>         - `env_kwargs`: *dict* 
>             - Default: {}
>     - Returns: None
>     - Description: Function which initialises an object of the class. Naming the object is the only required input. All other inputs are passed as keyword arguments that are used to create metadata or custom attributes or provide further functionality to custom methods.
>
>  * `_create_default_walls( )` 
>     - Accepts: None
>     - Returns: None
>     - Description: Generate walls to outline the arena based on the limits given in kwargs when initializing the object.
        Each wall is presented by a matrix
            [[xi, yi],
             [xf, yf]]
        where xi and yi are x y coordinates of one limit of the wall, and xf and yf are coordinates of the other limit.
        Walls are added to default_walls list, to then merge it with custom ones.
>
> * `_create_custom_walls()`
>     - Accepts: None
>     - Returns: None 
>     - Description: Custom walls method used to add new walls within the outlined boundary of the room. In this case it is empty since the environment is a simple square room. Override this method to generate more walls, see jupyter notebook with examples.
>  
> * `reset()`
>     - Accepts:
>         - `random_state`: *bool*
>             - Default: `False`
>             - Description: If True, sample a new position uniformly within the arena, use default otherwise.
>         - `custom_state`: *np.ndarray*
>             - Default: None
>             - Description: If given, use this array to set the initial state.
>     - Returns:
>         - `observation`: *ndarray*
>             - Description: Array of the observation of the agent in the environment. Because this is a fully observable environment, make_observation returns the state of the environment (Could be modified as the environments are evolves)
>         - `self.state`: *ndarray*
>             - Description: Vector of the x and y coordinate of the position of the animal in the environment (ndarray has shape (2,)).
>     - Description: Reset the environment variables and history.
>
> * `step()`
>     - Accepts:
>         - `action`: *ndarray (2,)*
>             - Default: `False`
>             - Description: Array containing the action of the agent, in this case the delta_x and detla_y increment to position.
>         - `normalize_step`: *bool*
>             - Default: None
>             - Description: If true, the action is normalized to have unit size, then scaled by the agent step size.
>     - Returns:
>         - `reward`: *float*
>             - Description: The reward that the agent receives for taking the given action in the current state
>         - `new_state`: *ndarray*
>             - Description: Updated state with the coordinates of the body position and head directions respectively.
>         - `observation`: *ndarray*
>             - Description: Array of the new observation of the agent in the environment (which is not necessarily the same as the state of the environment).
>     - Description: Given an action for the current state of the environment, runs the environment dynamics for the action and increases global counters. Then returns the new observation, new state and reward resulting from the action in that state.
>
> * `validate_action()`
>     - Accepts:
>         - `pre_state`: *ndarray (2,)*
>             - Description: 2d position of the agent in the environment before the action.
>         - `new_state`: *ndarray (2,)*
>             - Description: Potential 2d position of the agent in the environment after the action.
>     - Returns:
>         - `new_state`: *ndarray (2,)*
>             - Description: The corrected new state. If it is not crossing the wall, then the new_state stays the same, if the state cross the wall, new_state will be corrected to a valid place without crossing the wall.
>         - `crossed_wall`: *bool*
>             - Description: True if the change in state crossed a wall and was corrected.
>     - Description: Check if the new state is crossing any walls in the arena by taking an action. Corrects the new state if a wall is crossed.
>
> * `plot_trajectory()`
>     - Accepts:
>         - `history_data`: *list*
>             - Default: None
>             - Description: If None, it will use history data saved as attribute of the arena. Othwerwise use as a custom function holding list of interactions.
>         - `ax`: *mpl.axes._subplots.AxesSubplot (matplotlib axis from subplots)*
>             - Default: None
>             - Description: axis from subplot from matplotlib where the trajectory will be plotted.
>         - `return_figure`: *bool*
>             - Default: `False`
>             - Description: If true, it will return the figure variable generated to make the plot.
>         - `save_path`: *str*, *list of str* or *tuple of str*
>             - Default: `False`
>             - Description: The saving path of the generated figure, if None, no figure is saved.
>         - `plot_every`: *int*
>             - Default: 1
>             - Description: Number of time steps between plot points on the figure.  
>     - Returns:
>         - `ax`: *mpl.axes._subplots.AxesSubplot (matplotlib axis from subplots)*
>             - Description: Modified axis where the trajectory is plotted.
>         - `f`: *matplotlib.figure*
>             - Description: If return_figure parameters is True this figure will be returned.
>     - Description: Plot the Trajectory of the agent in the environment.
>
> * `render()`
>     - Accepts:
>         - `history_length`: *int*
>             - Default: 30
>             - Description: Number of time steps maintained in the render of the history. 
>     - Returns: None
>     - Description: Render the environment live through iterations.
