import numpy as np
from time import time
from multiprocessing import Pool
import inspect
import dill # nötig für 2.1.b
import networkx as nx


GREEN_CHECK = "\033[92m\u2714\033[0m"
RED_CROSS = "\033[91m\u2718\033[0m"
YELLOW_EXCLAMATION = "\033[93m\u26A0\033[0m"
BLUE_INFO = "\033[94mℹ\033[0m"


def test_task_2_1_a(die_class: object) -> None:
    """
    Test function for checking the implementation of the 'Die' class and its 'roll' method.

    Args:
        die_class (object): The class object to be tested.

    Returns:
        None
    """
    # check if class is named "Die"
    if die_class.__name__ != "Die":
        print(f"{RED_CROSS} Your class should be named 'Die', but got {die_class.__name__}")
        return
    
    # check if class has method "roll"
    if not hasattr(die_class, "roll") or not callable(getattr(die_class, "roll")):
        print(f"{RED_CROSS} Your class 'Die' does not have a method named 'roll'")
        return
    
    print(f"{GREEN_CHECK} Your class 'Die' and its method 'roll' look good.")


def wrapper_unserialize_method(serialized_method: bytes, kwargs: dict) -> callable:
    """
    Basically you can ignore this method. It's just a wrapper used in the method 'test_task_2_1_b'. But since you have already read this far: \n
    Executes a serialized method with the given keyword arguments.
    This is necassary because its very likely the method 'test_task_2_1_b' will be called from a Jupyter Notebook,
    which handles the 'main'-method differently than a normal Python script. So we need to serialize our method, wrap 
    it by deserializing it and then execute it in the desired thread. Also any refernce to classes not in PYTHONPATH won't be recognized,
    so make sure to follow the instructions listed in the 'test_task_2_1_b'-docstring! Holy moly, that's a lot of work for a simple test.

    Args:
        serialized_method (bytes): The serialized method to execute.
        kwargs (dict): The keyword arguments to pass to the method.

    Returns:
        callable: The result of executing the method with the given arguments.
    """
    method = dill.loads(serialized_method)
    return method(**kwargs)

def test_task_2_1_b(random_roll_method: callable, kwargs: dict, timeout: int = 180) -> None:
    """
    Test function for checking the implementation of the random roll method. \n
    ❗ - To use this function, your random_roll_method must have the die-class as a parameter. \n
    ❗ - Also the kwargs argument must contain the key 'die_class' pointing to the Die-Class. \n
    Also your random_roll_method should return a bool, wether the sum is reached or not. 
    Kills the Thread after <timeout> seconds if your method does not return.
    
    Example definition:
        random_roll_method(die_class: Die) -> bool
    Example usage:
        test_task_2_1_b(random_roll_method, {"die_class": Die}, timeout=180)

    Args:
        random_roll_method (callable): The random roll method to be tested. Read text above for more information.
        kwargs (dict): Arguments for the provided method. Read text above for more information.
        timeout (int, optional): Timeout value in seconds. Defaults to 180 (3 minutes).

    Returns:
        None
    """
    assert callable(random_roll_method), f"Expected a function, but got {type(random_roll_method)}"
    assert isinstance(kwargs, dict), f"Expected a dictionary (as arguments for the provided method), but got {type(kwargs)}"
    assert "die_class" in kwargs, f"The key 'die_class' in kwargs pointing to the Die-Class is required."
    assert "die_class" in inspect.signature(random_roll_method).parameters, "random_roll_method must have the parameter 'die_class'"
    assert isinstance(timeout, int), f"Expected an integer, but got {type(timeout)}"
    assert timeout > 0, f"Expected a positive integer, but got {timeout}"
    
    # check if random_roll_method has the 'Die' class as a parameter
    if "die_class" not in inspect.signature(random_roll_method).parameters:
        print(f"{RED_CROSS} Your random_roll_method should have the 'Die' class as a parameter with the name 'die_class'. Example: random_roll_method(die_class: Die) -> bool")
        return
    
    # check if kwargs contain the 'die_class' key
    if "die_class" not in kwargs:
        print(f"{RED_CROSS} The key 'die_class' in kwargs pointing to the Die-Class is required. Like this: kwargs = {{'die_class': Die}}")
        return
    
    pool = Pool(processes=1)
    serialilized_random_roll_method = dill.dumps(random_roll_method)
    
    try:
        result = pool.apply_async(wrapper_unserialize_method, (serialilized_random_roll_method, kwargs)).get(timeout=timeout)
        if not isinstance(result, bool):
            print(f"{RED_CROSS} Your random_roll_method should return a boolean value indicating wether the sum is reached or not, but got {type(result)}")
            pool.terminate()
            return
    except Exception as e:
        print(e)
        print(f"{YELLOW_EXCLAMATION} Your random_roll_method took too long (> {timeout} seconds) and was terminated or an exception occured.")
        pool.terminate()
        return
    
    pool.close()
    pool.join()
    pool.terminate()
    
    if result and isinstance(result, bool):
        print(f"{GREEN_CHECK} Your random_roll_method returns True if the sum is reached. Also below the 3 minutes timeout. Impressive!")
    else:
        print(f"{RED_CROSS} Your random_roll_method does not return True if the sum is reached. Please check your implementation.")
        
    
def test_task_2_1_c(*args) -> None:
    """
    This function is used to test the implementation of task 2.1.c.
    Since this task is not really testable, it will tell you that we will give you feedback once we have looked over your code.
    """
    print(f"{BLUE_INFO} This task is not really testable. So we will let you know if you have implemented the function correctly when we have looked over your code.")


def test_task_2_1_d(*args) -> None:
    """
    This function is used to test the implementation of task 2.1.d.
    Since this task is not really testable, it will tell you that we will give you feedback once we have looked over your code.
    """
    print(f"{BLUE_INFO} This task is not really testable. So we will let you know if you have implemented the function correctly when we have looked over your code.")


def test_task_2_1_e(evolutionary_dice_algorithm: callable, kwargs: dict) -> None:
    """
    This function is used to test the implementation of task 2.1.e. It's a function for evaluating the performance of your evolutionary dice algorithm.
    Gives information about the execution time and if the algorithm returns the expected result.

    Parameters:
    - evolutionary_dice_algorithm (callable): The evolutionary dice algorithm to be tested.
    - kwargs (dict): The arguments for the provided method.

    Returns:
    - float: The time it took to run the evolutionary dice algorithm.
    """

    assert callable(evolutionary_dice_algorithm), f"Expected a function, but got {type(evolutionary_dice_algorithm)}"
    assert isinstance(kwargs, dict), f"Expected a dictionary (as arguments for the provided method), but got {type(kwargs)}"

    start_time = time()
    result = evolutionary_dice_algorithm(**kwargs)
    end_time = time() - start_time
    
    # check if the function returns a value
    if result is None:
        print(f"{RED_CROSS} Your function does not return a value. Please return the best solution.\n{BLUE_INFO} All dice should show 6)")
        return

    if not (isinstance(result, list) or isinstance(result, np.ndarray)):
        print(f"{RED_CROSS} Your function should return a list or a numpy array of numbers, but got {type(result)}")
        return

    result = np.array(result)
    
    # check if the function returns roughly the expected result
    if np.sum(result) == 6 * len(result):
        print(f"{GREEN_CHECK} Your evolutionary algorithm rolls all dice to show 6: {result}")
    else:
        print(f"{RED_CROSS} Your evolutionary algorithm does not roll all dice to show 6: {result} ≠ {[6] * len(result)}")
    print(f"{BLUE_INFO} Your evolutionary algorithm took {end_time:.2f} seconds to run.")
    
    return end_time
    

def test_task_2_1_f(evolutionary_dice_algorithm: callable, evolutionary_dice_algorithm_kwargs: dict, 
                    optimized_evolutionary_dice_algorithm: callable, optimized_evolutionary_dice_algorithm_kwargs: dict) -> None:
    """
    This function is used to test the implementation of task 2.1.d. It's a function for comparing the performance of your two evolutionary dice algorithms.
    Gives information about the execution time and if the optimized algorithm is faster than the default one.

    Args:
        evolutionary_dice_algorithm (callable): The default evolutionary algorithm to be tested.
        evolutionary_dice_algorithm_kwargs (dict): Arguments for the default evolutionary algorithm.
        optimized_evolutionary_dice_algorithm (callable): The optimized evolutionary algorithm to be tested.
        optimized_evolutionary_dice_algorithm_kwargs (dict): Arguments for the optimized evolutionary algorithm.

    Returns:
        float: The factor by which the optimized evolutionary algorithm was faster than the default evolutionary algorithm.
    """
    assert callable(evolutionary_dice_algorithm), f"Expected a function, but got {type(evolutionary_dice_algorithm)}"
    assert isinstance(evolutionary_dice_algorithm_kwargs, dict), f"Expected a dictionary (as arguments for the provided method), but got {type(evolutionary_dice_algorithm_kwargs)}"
    assert callable(optimized_evolutionary_dice_algorithm), f"Expected a function, but got {type(optimized_evolutionary_dice_algorithm)}"
    assert isinstance(optimized_evolutionary_dice_algorithm_kwargs, dict), f"Expected a dictionary (as arguments for the provided method), but got {type(optimized_evolutionary_dice_algorithm_kwargs)}"
    
    # run both EAs
    print(f"{BLUE_INFO} Testing the default evolutionary algorithm...")
    default_end_time = test_task_2_1_e(evolutionary_dice_algorithm, evolutionary_dice_algorithm_kwargs)
    print(f"\n{BLUE_INFO} Testing the optimized evolutionary algorithm...")
    optimized_end_time = test_task_2_1_e(optimized_evolutionary_dice_algorithm, optimized_evolutionary_dice_algorithm_kwargs)
    # compare both results
    diff = default_end_time / optimized_end_time
    if diff <= 1:
        print(f"\n{YELLOW_EXCLAMATION} Your optimized evolutionary algorithm was not faster than your default evolutionary algorithm.")
    else:
        print(f"\n{GREEN_CHECK} Your optimized evolutionary algorithm was {diff:.2f} times faster than your default evolutionary algorithm.")
        
    return diff
        

def test_task_2_2_a(generate_graph: callable, n_nodes: int, n_edges: int, verbose = True) -> None:
    """
    Test function for task 2.2.a. It checks if your provided methods generated graph has the correct number of nodes and edges.

    Args:
        generate_graph (callable): A function that generates a graph.
        n_nodes (int): The number of nodes in the generated graph.
        n_edges (int): The number of edges in the generated graph.
        verbose (bool, optional): If True, the function will print the result of the test. Defaults to True.

    Returns:
        networkx.Graph: The generated graph.
    """
    assert callable(generate_graph), f"Expected a function, but got {type(generate_graph)}"
    assert isinstance(n_nodes, int), f"Expected an integer, but got {type(n_nodes)}"
    assert isinstance(n_edges, int), f"Expected an integer, but got {type(n_edges)}"
    
    graph = generate_graph(n_nodes, n_edges)
    # check if the function returns a value
    if not isinstance(graph, nx.Graph) and verbose:
        print(f"{RED_CROSS} Your function should return a networkx.Graph object, but got {type(graph)}")
        return None
    # check for false amount of edges and nodes
    if len(graph.nodes) != n_nodes and verbose:
        print(f"{RED_CROSS} Your graph should have {n_nodes} nodes, but got {len(graph.nodes)}")
    if len(graph.edges) != n_edges and verbose:
        print(f"{RED_CROSS} Your graph should have {n_edges} edges, but got {len(graph.edges)}")
    # check for correct amount of edges and nodes
    if len(graph.nodes) == n_nodes and len(graph.edges) == n_edges and verbose:
        print(f"{GREEN_CHECK} Your function generates a graph with {n_nodes} nodes and {n_edges} edges.")
    
    return graph


def test_task_2_2_b(*args) -> None:
    """
    This function is used to test the implementation of task 2.2.b.
    Since this task is not really testable, it will tell you that we will give you feedback once we have looked over your code.
    """
    print(f"{BLUE_INFO} This task is not really testable. So we will let you know if you have implemented the function correctly when we have looked over your code.")
    
    
def test_task_2_2_c(evolutionary_clique_algorithm: callable, kwargs: dict) -> None:
    """
    Test function for task 2.2.c. It checks if your evolutionary clique algorithm finds one of the maximum cliques in the provided graph. Your provided method should return only the selected Node-IDs (int) that supposed to form a clique.\n
    ❗ - The provided method must have the argument 'gragh' (networkx.graph). \n
    ❗ - The kwargs argument must conatin the key 'gragh' pointing to the networkx.graph object.

    Args:
        evolutionary_clique_algorithm (callable): The evolutionary clique algorithm to test.
        kwargs (dict): Dictionary of arguments for the evolutionary clique algorithm. MUST conatin the key 'graph' pointing to the networkx.Graph object.

    Returns:
        object: The result of your evolutionary clique algorithm.
    """
    assert callable(evolutionary_clique_algorithm), f"Expected a function, but got {type(evolutionary_clique_algorithm)}"
    assert isinstance(kwargs, dict), f"Expected a dictionary (as arguments for the provided method), but got {type(kwargs)}"
    assert "graph" in kwargs, f"The key 'graph' in kwargs pointing to the networkx.Graph object is required."
    assert isinstance(kwargs["graph"], nx.Graph), f"The value of the key 'graph' in kwargs should be a networkx.Graph object, but got {type(kwargs['graph'])}"
    assert "graph" in inspect.signature(evolutionary_clique_algorithm).parameters, "evolutionary_clique_algorithm must have the parameter 'graph'"
    
    # get maximum cliques
    graph = kwargs["graph"]
    if graph is None:
        print(f"{RED_CROSS} Somehow your graph is None. Please check your implementation.")
        return None
    cliques = list(nx.find_cliques(graph))
    maximum_cliques = [sorted(clique) for clique in cliques if len(clique) == max(len(c) for c in cliques)]
    
    # run algorithm
    start_time = time()
    result = evolutionary_clique_algorithm(**kwargs)
    end_time = time() - start_time
    # check if the function returns a typewise correct result
    if result is None or (not isinstance(result, list) or isinstance(result, np.ndarray)):
        print(f"{RED_CROSS} Your function should return a list or a numpy array of integers (Node-IDs), but got {type(result)}")
        return None
    if isinstance(result, np.ndarray):
        result = result.tolist()
    result = sorted(result)
    
    # check the result
    if len(result) != len(maximum_cliques[0]):
        print(f"{RED_CROSS} Your evolutionary algorithm did not find one of the maximum cliques. Expected one of: {maximum_cliques}, but got {result}")
    elif result in maximum_cliques:
        print(f"{GREEN_CHECK} Your evolutionary algorithm found one of the maximum cliques: {result}\n all possible maximum cliques are: {maximum_cliques}")
        print(f"{BLUE_INFO} Your evolutionary algorithm took {end_time:.2f} seconds to run.")
    else:
        print(f"{RED_CROSS} Your evolutionary algorithm did not find one of the maximum cliques. Expected one of: {maximum_cliques}, but got {result}")
        
    return result


def test_task_2_2_d(*args) -> None:
    """
    This function is used to test the implementation of task 2.2.d.
    Since this task is not really testable, it will tell you that we will give you feedback once we have looked over your code.
    """
    print(f"{BLUE_INFO} This task is not really testable. So we will let you know if you have implemented the function correctly when we have looked over your code.")
    
    
def test_task_2_2_e(evolutionary_clique_algorithm: callable, kwargs: dict) -> None:
    """
    Test function for task 2.2.e. Checks for the Graphs (10,20), (10,40), (20,40), (20,150), (40,80), (40,200), (40,600). Your provided method should return only the selected Node-IDs (int) that supposed to form a clique.\n
    ❗ - The provided method must have the argument 'gragh' (networkx.graph). \n
    ❗ - Also the kwargs argument must conatin the key 'gragh' pointing to any networkx.graph object.

    Args:
        evolutionary_clique_algorithm (callable): The evolutionary clique algorithm to test.
        kwargs (dict): Dictionary of arguments for the evolutionary clique algorithm. MUST conatin the key 'graph' pointing to the networkx.Graph object.

    Returns:
        object: The result of your evolutionary clique algorithm.
    """
    assert callable(evolutionary_clique_algorithm), f"Expected a function, but got {type(evolutionary_clique_algorithm)}"
    assert "graph" in inspect.signature(evolutionary_clique_algorithm).parameters, "evolutionary_clique_algorithm must have the parameter 'graph'"
    
    configs = [(10,20), (10,40), (20,40), (20,150), (40,80), (40,200), (40,600)]
    
    for n_nodes, n_edges in configs:
        print(f"\n{BLUE_INFO} Testing graph with {n_nodes} nodes and {n_edges} edges...")
        graph = test_task_2_2_a(nx.gnm_random_graph, n_nodes, n_edges, verbose=False)
        kwargs["graph"] = graph
        result = test_task_2_2_c(evolutionary_clique_algorithm, kwargs)
        if result is None:
            print(f"{RED_CROSS} Your evolutionary algorithm did not return a result.")
