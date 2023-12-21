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


if __name__ == "__main__":
    test_q1("test_cases/q2/food_price1")