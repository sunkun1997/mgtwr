# mgtwr

To fit geographically weighted model, geographically and temporally weighted regression model and multiscale geographically and temporally weighted regression model. You can
read example.ipynb to know how to use it.

# model.py improve
When trying to use **parallel processing**, **model.py** is the function that is called. The original code was using multiprocessing.So based on cProfile and the pstats library ran statistics, it is found **thread.lock** to be the main time consuming task,which causes parallel processing to be slower than non-parallel processing. So attempts were made to compare the processing of ThreadPoolExecutor from concurrent.futures and joblib, and ultimately joblib was found to be the fastest case.All of the result was run on the provided **example.csv**.
As you can see, joblib greatly reduces the parallel processing time, and also has a large improvement compared to orginal, however, it should be noted that **{method 'acquire' of '_thread.lock' objects}** is still the most time-consuming task, and how to solve this is beyond my ability to do.

Related computer configuration:
12t Gen_Intel(R) Core(TM) i5-12600KF(6+4 for core 16 of threads)
Crucial 16GB DDR5-4800 UDIMM
(My computer configuration isn't that low also so I'm very sad why I have to run for 11 minutes when it's only 6 minutes in the example)

# result comparison:
1. Orginal(**thread=1**)
**time cost: 0:11:3.748**
         636805417 function calls (613137518 primitive calls) in 663.749 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
  2151360  169.201    0.000  225.785    0.000 d:\anaconda\Lib\site-packages\scipy\linalg\_basic.py:40(solve)
19328497/15022891   35.541    0.000   81.668    0.000 {built-in method numpy.core._multiarray_umath.implement_array_function}

2. Original(thread=15)
**time cost: 0:24:29.669**
         6022657 function calls (6019718 primitive calls) in 1469.671 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    23663 1397.118    0.059 1397.118    0.059 {method 'acquire' of '_thread.lock' objects}

3. ThreadPoolExecutor(**thread=15**)
**time cost: 0:07:44.877**
         99635100 function calls (99632161 primitive calls) in 464.878 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
 10310899  409.240    0.000  409.240    0.000 {method 'acquire' of '_thread.lock' objects}

4. joblib(**thread=15**)
time cost: 0:03:40.609
         12395230 function calls (12381086 primitive calls) in 220.610 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   902982  187.731    0.000  187.731    0.000 {method 'acquire' of '_thread.lock' objects}
     1245   22.471    0.018  215.907    0.173 d:\anaconda\Lib\site-packages\joblib\parallel.py:960(retrieve)
   403539    2.014    0.000  192.678    0.000 d:\anaconda\Lib\concurrent\futures\_base.py:428(result)
     1220    1.293    0.001  215.684    0.177 C:\Users\34456\AppData\Roaming\Python\Python311\site-packages\mgtwr\model.py:450(cal_aic)
