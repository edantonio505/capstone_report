from paramiko import SSHClient
from scp import SCPClient
import paramiko
 
def progress(filename, size, sent):
    print filename + " " + str(size) + " " + str(sent)
 

if __name__ == "__main__":
 	
 	ssh = SSHClient()
	ssh.load_system_host_keys()
	ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	ssh.connect('45.79.77.5', port=22, username="", password="")

	with open("../sec_files.txt") as files:
		for file in files:
			print file
		   
		 
		    # SCPCLient takes a paramiko transport as its only argument
		    # Just a no-op. Required sanitize function to allow wildcards.
			scp = SCPClient(ssh.get_transport(), sanitize=lambda x: x, progress=progress)
			# with SCPClient(ssh.get_transport()) as scp:
		    # scp.listdir("/var/tmp")
			scp.get(file)
