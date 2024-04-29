import subprocess

def main():
    for i in range(4):
        subprocess.run(
            ["bash", "execute.bash"], 
            env={
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "8000",
                "LOCAL_RANK": f"{i}",
            }
)