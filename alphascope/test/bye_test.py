from alphascope.src import byeworld as bw


def test_bye_world_no_params():
    assert bw.say_bye() == "bye, world"


def test_hello_world_with_params():
    assert bw.say_bye("felicia") == "bye, felicia"
