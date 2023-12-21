import os

def get_all_test_names(question):
    # get all test names from test_cases folder
    test_names = []
    for root, dirs, files in os.walk(f"test_cases/{question}"):
        for file in files:
            if file.endswith(".test"):
                test_names.append(os.path.join(root, file[:-5]))
    return test_names

def pytest_generate_tests(metafunc):
    if "q1testname" in metafunc.fixturenames:
        test_names = get_all_test_names("q1")
        metafunc.parametrize("q1testname", test_names)
    if "q2testname" in metafunc.fixturenames:
        test_names = get_all_test_names("q2")
        metafunc.parametrize("q2testname", test_names)
    if "q3testname" in metafunc.fixturenames:
        test_names = get_all_test_names("q3")
        metafunc.parametrize("q3testname", test_names)
    if "q4testname" in metafunc.fixturenames:
        test_names = get_all_test_names("q4")
        metafunc.parametrize("q4testname", test_names)
    if "q5testname" in metafunc.fixturenames:
        test_names = get_all_test_names("q5")
        metafunc.parametrize("q5testname", test_names)
    if "q6testname" in metafunc.fixturenames:
        test_names = get_all_test_names("q6")
        metafunc.parametrize("q6testname", test_names)
    if "q7testname" in metafunc.fixturenames:
        test_names = get_all_test_names("q7")
        metafunc.parametrize("q7testname", test_names)
    if "q8testname" in metafunc.fixturenames:
        test_names = get_all_test_names("q8")
        metafunc.parametrize("q8testname", test_names)

if __name__ == "__main__":
    print(get_all_test_names("q8"))