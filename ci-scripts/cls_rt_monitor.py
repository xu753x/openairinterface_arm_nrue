#/*
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
# Real Time Stats Monitoring on google sheets and charts 
#
#   Required Python Version
#     Python 3.x
#
#---------------------------------------------------------------------

#-----------------------------------------------------------
# Import
#-----------------------------------------------------------

 
#import google spreadsheet API
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import re


import datetime   #now() and date formating
from datetime import datetime


#-----------------------------------------------------------
# Outside Class Functions 
#-----------------------------------------------------------


def createChartBody(sheetId, chartType, chartTitle, xTitle, yTitle, startRowIndex, endRowIndex, startColumnIndex, endColumnIndex):
  #this processing is specific to the chart we want to build
  #reason why it is not in the class itself 
  body = {
  "requests": [
    {
      "addChart": {
        "chart": {
          "spec": {
            "title": chartTitle.upper(),
            "titleTextFormat": {
            "bold": True,
            "fontSize": 24
            },
            "basicChart": {
              "chartType": chartType,
              "legendPosition": "BOTTOM_LEGEND",
              "axis": [
                {
                  "position": "BOTTOM_AXIS",
                  "title": xTitle,
                  "format": {
                  "bold": True,
                  "fontSize": 24
                   }
                },
                {
                  "position": "LEFT_AXIS",
                  "title": yTitle,
                  "format": {
                  "bold": True,
                  "fontSize": 24
                  }
                }
              ],
              "domains": [
                {
                  "domain": {
                    "sourceRange": {
                      "sources": [
                        {
                          "sheetId": sheetId,
                          "startRowIndex": startRowIndex,
                          "endRowIndex": endRowIndex,
                          "startColumnIndex": 0,
                          "endColumnIndex": 1
                        }
                      ]
                    }
                  }
                }
              ],
              "series": [
                {
                  "series": {
                    "sourceRange": {
                      "sources": [
                        {
                          "sheetId": sheetId,
                          "startRowIndex": startRowIndex,
                          "endRowIndex": endRowIndex,
                          "startColumnIndex": startColumnIndex,
                          "endColumnIndex": endColumnIndex
                        }
                      ]
                    }
                  },
                  "targetAxis": "LEFT_AXIS"
                },
                {
                  "series": {
                    "sourceRange": {
                      "sources": [
                        {
                          "sheetId": sheetId,
                          "startRowIndex": startRowIndex,
                          "endRowIndex": endRowIndex,
                          "startColumnIndex": startColumnIndex+1,
                          "endColumnIndex": endColumnIndex+1
                        }
                      ]
                    }
                  },
                  "targetAxis": "LEFT_AXIS"
                }
              ],
              "headerCount": 1
            }
          },
          "position": {
            "newSheet": True
          }
        }
      }
    }
  ]
}
  return body



def build_RT_Row(Branch,Commit,keys,filename):
    #this processing is specific to the row we want to build
    #reason why it is not in the class itself
    
    row=[]
        
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M")	
    row.append(dt_string)

    row.append(Branch)
    row.append(Commit)    

    real_time_stats = {}
    f=open(filename,"r")
    for line in f.readlines():
        for k in keys:
            result = re.search(k, line)     
            if result is not None:
                tmp=re.match(rf'^.*?(\b{k}\b.*)',line.rstrip()) #from python 3.6 we can use literal string interpolation for the variable k, using rf' in the regex 
                real_time_stats[k]=tmp.group(1)       
    f.close()

    for k in keys:
      tmp=re.match(r'^(?P<metric>.*):\s+(?P<avg>\d+\.\d+) us;\s+\d+;\s+(?P<max>\d+\.\d+) us;',real_time_stats[k])
      if tmp is not None:
          metric=tmp.group('metric')
          avg=tmp.group('avg')
          max=tmp.group('max')
          row.append(float(avg))
          row.append(float(max))
    
    if len(row)==3: #if row was not updated (missing data for ex), then return an empty row
        row=[]
    return row

#-----------------------------------------------------------
# gSheet Class 
#-----------------------------------------------------------

#google Sheets class with main util functions
#self.gChart calls createChartBody function outside of the class that is specific to our application, 
#possible enhancement : we could use function pointers if we had a need for several kind of charts

class gSheet:
    def __init__(self, creds_file, spreadsheet, worksheet, columns, row, col): 
        #authorization
        self.scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
        self.creds = ServiceAccountCredentials.from_json_keyfile_name(creds_file, self.scope)
        self.client = gspread.authorize(self.creds)
        #spreadsheet
        self.ss = self.client.open(spreadsheet)
        #worksheet
        try: #try to create a sheet, but it may exist already
            self.sheet = self.ss.add_worksheet(title=worksheet, rows=row, cols=col) #create a new one
            #create header
            self.createHeader(columns)
        except: #if exists just open it, disregard sizing (col and row)
            self.sheet = self.ss.worksheet(worksheet)


    def existSheet(self, worksheet):
        #check if a worksheet exists already in the spreadsheet
        #could not find an API function to do that
        sheets_list = self.ss.worksheets()
        sheets_name =[]
        for i in range(0,len(sheets_list)):
            sheets_name.append(sheets_list[i].title)
        if worksheet in sheets_name:
            return True
        else:
            return False

        
    def createHeader(self, columns):
        #basically create row 1 with column names
        self.sheet.insert_row(columns, index=1, value_input_option='RAW')


    def insertRow(self, row):
        #insert row always on top, so that the most recent record is on top, and the oldest at the bottom          
        self.sheet.insert_row(row, index=2, value_input_option='RAW')
        

    def renameSheet(self,worksheet,newName):
        #to rename a worksheet
        sheetId = self.ss.worksheet(worksheet)._properties['sheetId']
        body = {
                  "requests": [
                    {
                    "updateSheetProperties": {
                            "properties": {
                                    "sheetId": sheetId,
                                    "title": newName
                                    },
                            "fields": "title",
                            }
                        }
                    ]
                  }
        self.ss.batch_update(body) 



    def gChart(self, worksheet, chartName, chartType, xTitle , yTitle, startRowIndex, endRowIndex, startColumnIndex, endColumnIndex):

        if self.existSheet(chartName)==True:  
            #print("Deleting existing sheet (Chart) : " + chartName)
            self.sheet = self.ss.worksheet(chartName)
            self.ss.del_worksheet(self.sheet) #start by deleting the old sheet
       
            
        #print("Creating new chart")
        self.sheet = self.ss.worksheet(worksheet)

        
        #create the Chart by applying the body request
        sheetId = self.ss.worksheet(worksheet)._properties['sheetId']
        body=createChartBody(sheetId, chartType, chartName, xTitle, yTitle, startRowIndex, endRowIndex, startColumnIndex, endColumnIndex)
        self.ss.batch_update(body)
        
        #rename the Sheet "Chartxx" to its column name
        #print("Renaming new chart")
        sheets_list = self.ss.worksheets()
        for i in range(0,len(sheets_list)):
            tmp = re.match("^(Chart\d+)$", sheets_list[i].title)
            if tmp is not None:
                self.renameSheet(tmp.group(1),chartName)
            
        
def gNB_RT_monitor(ranBranch, ranCommitID, eNBlogFile):

  ########################
	#data log to google sheet
	#########################
	keys=['feprx','feptx_prec','feptx_ofdm','feptx_total','L1 Tx processing','DLSCH encoding','L1 Rx processing','PUSCH inner-receiver','PUSCH decoding']
	columns=["Date Time","Branch","Commit",\
			'feprx avg','feprx max',\
			'feptx_prec avg','feptx_prec max',\
			'feptx_ofdm avg','feptx_ofdm max',\
			'feptx_total avg','feptx_total max',\
			'L1 Tx processing avg','L1 Tx processing max',\
			'DLSCH encoding avg','DLSCH encoding max',\
			'L1 Rx processing avg','L1 Rx processing max',\
			'PUSCH inner-receiver avg','PUSCH inner-receiver max',\
			'PUSCH decoding avg','PUSCH decoding max']

	#open gsheet
	gRT=gSheet("/home/oaicicd/ci_gsheet_creds.json", 'RealTime Monitor', 'timeseries', columns, 10000, 50)
	#build row, but insert row and update charts only if data were found
	row=build_RT_Row(ranBranch,ranCommitID,keys,eNBlogFile)
	if len(row)!=0:
		gRT.insertRow(row)
		#updating charts
		#will plot avg and max on the same chart, these 2 columns have to be side by side
		#spreadsheet , chart name, chart type, x title , y title, start row inc header, end row, start col, end col
		gRT.gChart('timeseries', 'feprx', 'COLUMN', 'CI RUNs DateTime' , 'ProcessingTime (us)' , 0, 1000, 3, 4)
		gRT.gChart('timeseries', 'feptx_prec', 'COLUMN', 'CI RUNs DateTime' , 'ProcessingTime (us)' , 0, 1000, 5, 6)
		gRT.gChart('timeseries', 'feptx_ofdm', 'COLUMN', 'CI RUNs DateTime' , 'ProcessingTime (us)' , 0, 1000, 7, 8)
		gRT.gChart('timeseries', 'feptx_total', 'COLUMN', 'CI RUNs DateTime' , 'ProcessingTime (us)' , 0, 1000, 9, 10)
		gRT.gChart('timeseries', 'L1 Tx proc', 'COLUMN', 'CI RUNs DateTime' , 'ProcessingTime (us)' , 0, 1000, 11, 12)
		gRT.gChart('timeseries', 'DLSCH enc', 'COLUMN', 'CI RUNs DateTime' , 'ProcessingTime (us)' , 0, 1000, 13, 14)
		gRT.gChart('timeseries', 'L1 Rx proc', 'COLUMN', 'CI RUNs DateTime' , 'ProcessingTime (us)' , 0, 1000, 15, 16)
		gRT.gChart('timeseries', 'PUSCH inner-rec', 'COLUMN', 'CI RUNs DateTime' , 'ProcessingTime (us)' , 0, 1000, 17, 18)
		gRT.gChart('timeseries', 'PUSCH dec', 'COLUMN', 'CI RUNs DateTime' , 'ProcessingTime (us)' , 0, 1000, 19, 20)
	  ########################
        
    

##################################
#USAGE EXAMPLES
#################################
    
#    keys=['feprx','feptx_prec','feptx_ofdm','feptx_total','L1 Tx processing','DLSCH encoding','L1 Rx processing','PUSCH inner-receiver','PUSCH decoding']
#    columns=["Date Time","Branch","Commit",\
#             'feprx avg','feprx max',\
#             'feptx_prec avg','feptx_prec max',\
#             'feptx_ofdm avg','feptx_ofdm max',\
#             'feptx_total avg','feptx_total max',\
#             'L1 Tx processing avg','L1 Tx processing max',\
#             'DLSCH encoding avg','DLSCH encoding max',\
#             'L1 Rx processing avg','L1 Rx processing max',\
#             'PUSCH inner-receiver avg','PUSCH inner-receiver max',\
#             'PUSCH decoding avg','PUSCH decoding max']

#    #creation or opening
#    gRT=gSheet("creds.json", 'RealTime Monitor', 'timeseries', columns, 10000, 50)
#    #an other example
#    gmy=gSheet("creds.json", 'RealTime Monitor', 'titi', ['a','b','c'], 10, 10)
    
    
#    #inserting rows
#    row=build_RT_Row("my test branch 2","123456789",keys,"ci_scripts/enb_040000.log")
#    gRT.insertRow(row)
    
#    #updating charts
#    #will plot avg and max on the same chart, these 2 columns have to be side by side
#    #spreadsheet , chart name, chart type, x title , y title, start row inc header, end row, start col, end col excluded
#    gRT.gChart('timeseries', 'feprx', 'COLUMN', 'CI RUNs DateTime' , 'ProcessingTime (us)' , 0, 1000, 3, 4)




