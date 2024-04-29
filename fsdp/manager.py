import subprocess

def main():
    for i in range(1, 4):
        subprocess.Popen(
            ["bash", "execute.bash"], 
            env={
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "8000",
                "LOCAL_RANK": f"{i}",
                "HOME": "/workspace"
            },
            capture_output=False
        )
    
    subprocess.run(
        ["bash", "execute.bash"], 
        env={
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "8000",
            "LOCAL_RANK": "0",
            "HOME": "/workspace"
        }
    )

if __name__ == "__main__":
    main()
