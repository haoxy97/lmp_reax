import array
import time

def test_stack_performance(n):
    # Stack operations with Python list
    start_time_list = time.time()
    stack_list = []
    for i in range(n):
        stack_list.append(i)
    for _ in range(n):
        stack_list.pop()
    end_time_list = time.time()
    list_time = end_time_list - start_time_list

    # Stack operations with array.array
    start_time_array = time.time()
    stack_array = array.array('i')
    for i in range(n):
        stack_array.append(i)
    for _ in range(n):
        stack_array.pop()
    end_time_array = time.time()
    array_time = end_time_array - start_time_array

    return list_time, array_time

# Testing with 1 million elements
n_elements = 1_000_000
list_perf, array_perf = test_stack_performance(n_elements)

print(f"List performance: {list_perf:.6f} seconds")
print(f"Array performance: {array_perf:.6f} seconds")