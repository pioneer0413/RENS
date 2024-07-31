import pyRAPL

pyRAPL.setup() 

@pyRAPL.measureit
def foo(scale):
    total = 0
    for i in range(scale):
        total += i

foo()

print("DONE")