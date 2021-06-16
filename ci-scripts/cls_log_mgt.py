# * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
# * contributor license agreements.  See the NOTICE file distributed with
# * this work for additional information regarding copyright ownership.
# * The OpenAirInterface Software Alliance licenses this file to You under
# * the OAI Public License, Version 1.1  (the "License"); you may not use this file
# * except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.openairinterface.org/?page_id=698
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *-------------------------------------------------------------------------------
# * For more information about the OpenAirInterface (OAI) Software Alliance:
# *      contact@openairinterface.org
# */
#---------------------------------------------------------------------
#
#   Required Python Version
#     Python 3.x
#
#---------------------------------------------------------------------

#import sys
import re
import subprocess

class Log_Mgt:

	def __init__(self,IPAddress,Password,path,filesize):
		self.IPAddress=IPAddress
		self.Password="oaicicd"
		self.path=path
		self.filesize=filesize

#-----------------$
#PRIVATE# Methods$
#-----------------$


	def __CheckAvailSpace(self):
		HOST=self.IPAddress
		COMMAND="df "+ self.path
		ssh = subprocess.Popen(["ssh", "%s" % HOST, COMMAND],shell=False,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
		result = ssh.stdout.readlines()
		s=result[1].decode('utf-8').rstrip()
		tmp=s.split()
		return tmp[3]

	def __GetOldestFile(self):
		HOST=self.IPAddress
		COMMAND="ls -rtl "+ self.path
		ssh = subprocess.Popen(["ssh", "%s" % HOST, COMMAND],shell=False,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
		result = ssh.stdout.readlines()
		s=result[1].decode('utf-8').rstrip()
		tmp=s.split()
		return tmp[8]

#-----------------$
#PUBLIC Methods$
#-----------------$



	def LogRotation(self):
		avail_space = self.__CheckAvailSpace()
		print("Avail Space : " + avail_space + " / Artifact Size : " + self.filesize)
		if filesize > avail_space:
			oldestfile=self.__GetOldestFile()
			HOST=self.IPAddress
			COMMAND="echo " + self.Password + " | sudo -S rm "+ self.path + "/" + oldestfile
			print(COMMAND)
			ssh = subprocess.Popen(["ssh", "%s" % HOST, COMMAND],shell=False,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
		else:
			print("Still some space left for artifacts storage")
			

#if __name__ == "__main__":
#	IPAddress=sys.argv[1]
#	Password=sys.argv[2]
#	path=sys.argv[3]
#	filesize=sys.argv[4]
#	log=Log_Mgt(IPAddress,Password,path,filesize)
#	log.LogRotation()
