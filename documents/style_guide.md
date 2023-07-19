
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

> model_name : str
>         The name of the new agent class you are implementing. Can be any valid string and will usually be used for labelling plots or printing to 			   	    terminal.
> mod_kwargs: dict
>         Dictionary of keyword arguments passed to the "\__init__()" function during instantiation of the object.
> metadata
>         Dictionary reserved for containing specific description details for each model. By default it just captures all keyword arguments passed in during 	    instantiation of a new object.
> obs_history: list
>         List of the agent's past observations obtained while interacting with the environment. This is populated progressively with each call of the act                  		method.
> global_steps: int
>         Records the number of updates done to the weights or parameters of a model implemented within the class *if* one is used.

Additionally the class will also inherit the necessary methods that the rest of the library will use to interface with the its objects. These are as follows:

> \_\_init\_\_( )
> 	Accepts: 
> 		model_name : str
> 			Default: "default_model"
> 		**mod_kwargs: dict
> 			Default: {}
> 	Returns: None
> 	Description: Function which initialises an object of the class. Naming the object is the only required input. All other inputs are passed as keyword 	arguments that are used to create metadata or custom attributes or provide further functionality to custom methods.
>
> reset() <!-- in the code the act function populates obs_history but this doesn't reset it -->
> 	Accepts:
> 		None
> 	Returns:
> 		None
> 	Description: Erases all memory from the model, re-initialises all relevant parameters and builds the original object from scratch.
>
> neural_response() <!-- I still think this needs to return something. The base_class needs to work with the full 												pipeline -->
> 	Accepts:
> 		None
> 	Returns:
> 		None <!-- Can't be none if we say it will be compared to experimental data -->
> 	Description: Returns the neural representation of the model performing the given task. Output will be compared against real experimental data.
>
> act()	
> 	Accepts:		
> 		obs: np.array <!-- Is this too specific? -->
> 			Default: None 
> 			Description: Observation from the environment class needed to choose the right action
> 		policy_func: func <!-- Check how to write func as a type -->
> 			Default: None 
> 			Description: Arbitrary function that represents a custom policy that receives and observation and gives an action
> 	Returns:
> 		action: float <!-- Its a float of shape 2 though, so how should I write that? -->
> 			Description: The action value indicating the direction the agent moves in 2d space
> 	Description: Chooses and executes and action for the agent. Typically depends on the agent's observations of the environment.
>
> update()	
> 	Accepts:
> 		None
> 	Returns:
> 		None
> 	Description: Alters the parameters of the model (if there are any) likely based on the observation history to simulate learning.
>
> save_agent()
> 	Accepts:
> 		save_path: str
> 			Default: None 
> 			Description: Path to the file where the objects state and information will be saved
> 	Returns:
> 		None
> 	Description: Saves the current state and object information to be able to re-instantiate the environment from scratch.
>
> restore_agent(self, save_path: str)	
> 	Accepts:
> 		save_path: str <!-- bad variable name -->
> 			Default: None 
> 			Description: Path to the file where the objects state and information will be restored from.
> 	Returns:
> 		None
> 	Description: Restores and re-instantiate the environment from scratch using the state and object information stored in the file at "save_path".
>
> \_\_eq\_\_(self, other) <!-- check what this does -->
> 	Accepts:
> 		other: dict
> 			Default: None
> 			Description: <!-- todo -->
> 	Returns:
> 		bool <!-- do I just name the variable here? -->
> 			Description: True if dictionaries are the same, False if they are different.
> 	Description: Determines whether two dictionaries are the same or equal.

