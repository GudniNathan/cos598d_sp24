import subprocess

def main():
    subprocess.run(["git", "pull"])
    
    processes = []
    for i in range(1, 4):
        p = subprocess.Popen(
            ["bash", "execute.bash"], 
            env={
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "8000",
                "LOCAL_RANK": f"{i}",
                "HOME": "/workspace"
            },
            #stdout=subprocess.DEVNULL, # suppress output
            #stderr=subprocess.DEVNULL
        )
        processes.append(p)
    
    subprocess.run(
        ["bash", "execute.bash"], 
        env={
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "8000",
            "LOCAL_RANK": "0",
            "HOME": "/workspace"
        },
    )
    for p in processes:
        p.wait() 

if __name__ == "__main__":
    main()
