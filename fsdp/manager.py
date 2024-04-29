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
            stdout=subprocess.DEVNULL, # suppress output
            stderr=subprocess.DEVNULL
        )
    
    subprocess.run(
        ["bash", "execute.bash"], 
        env={
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "8000",
            "LOCAL_RANK": "0",
            "HOME": "/workspace"
        },
        capture_output=True
    )

if __name__ == "__main__":
    main()
