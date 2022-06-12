from alphascope.src import helloworld as hw
from alphascope.src import byeworld as bw


def first_test():
    return hw.say_hello("ak")


def last_test():
    return bw.say_bye("ak")


print(first_test())
print(last_test())
