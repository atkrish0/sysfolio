from alphascope.src import helloworld as hw


def test_hello_world_no_params():
    assert hw.say_hello() == "hello, world"


def test_hello_world_with_params():
    assert hw.say_hello("ak") == "hello, ak"
