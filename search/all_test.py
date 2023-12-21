import subprocess

python_cmd = 'python3 autograder.py'

def test_q1(q1testname):
    command = f"{python_cmd} -t {q1testname}"

    p = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = [line for line in iter(p.stdout.readline, b'') if b"PASS" in line]
    print("total lines =", len(lines))
    assert len(lines) > 0

def test_q2(q2testname):
    command = f"{python_cmd} -t {q2testname}"

    p = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = [line for line in iter(p.stdout.readline, b'') if b"PASS" in line]
    print("total lines =", len(lines))
    assert len(lines) > 0

def test_q3(q3testname):
    command = f"{python_cmd} -t {q3testname}"

    p = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = [line for line in iter(p.stdout.readline, b'') if b"PASS" in line]
    print("total lines =", len(lines))
    assert len(lines) > 0

def test_q4(q4testname):
    command = f"{python_cmd} -t {q4testname}"

    p = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = [line for line in iter(p.stdout.readline, b'') if b"PASS" in line]
    print("total lines =", len(lines))
    assert len(lines) > 0

def test_q5(q5testname):
    command = f"{python_cmd} -t {q5testname}"

    p = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = [line for line in iter(p.stdout.readline, b'') if b"PASS" in line]
    print("total lines =", len(lines))
    assert len(lines) > 0

def test_q6(q6testname):
    command = f"{python_cmd} -t {q6testname}"

    p = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = [line for line in iter(p.stdout.readline, b'') if b"PASS" in line]
    print("total lines =", len(lines))
    assert len(lines) > 0

def test_q7(q7testname):
    command = f"{python_cmd} -t {q7testname}"

    p = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = [line for line in iter(p.stdout.readline, b'') if b"PASS" in line]
    print("total lines =", len(lines))
    assert len(lines) > 0

def test_q8(q8testname):
    command = f"{python_cmd} -t {q8testname}"

    p = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = [line for line in iter(p.stdout.readline, b'') if b"PASS" in line]
    print("total lines =", len(lines))
    assert len(lines) > 0

if __name__ == "__main__":
    test_q1("test_cases/q2/food_price1")