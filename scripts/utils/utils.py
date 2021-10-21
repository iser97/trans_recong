from multiprocess.dummy import Pool as ThreadPool


def map(func, input, thread_num=5):
    with ThreadPool(thread_num) as P:
        res = P.map(func, input)
    return res