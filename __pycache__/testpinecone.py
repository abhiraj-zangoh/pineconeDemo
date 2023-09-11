import pinecone
import time
# Initialize Pinecone
pinecone.init(api_key="acc66926-3691-46f8-baa2-20539770fa16", environment="gcp-starter")

index_name = "example-index"

# Delete the index if it exists (optional)

# pinecone.delete_index(index_name)

# Create the index
# pinecone.create_index(index_name, dimension=1, metric="euclidean")
index = pinecone.Index("example-index")
json_data=[
  {
    "lastName": "Aaron",
    "middleInitial": "D",
    "firstName": "Kareem",
    "jobClass": "UTILITIES INST REPAIR II",
    "agencyName": "Public Works - Water & Waste Water",
    "agencyID": "A50",
    "annualSalary": 34115,
    "grossPay": 44557.99,
    "hireDate": "2018-08-27",
    "fiscalYear": "FY2021",
    "ObjectId": 1
  },
  {
    "lastName": "Aaron",
    "middleInitial": "R",
    "firstName": "Lynda",
    "jobClass": "ELECTION JUDGES REGULAR",
    "agencyName": "D01",
    "agencyID": "D01",
    "annualSalary": 0,
    "grossPay": 185,
    "hireDate": "2020-12-08",
    "fiscalYear": "FY2021",
    "ObjectId": 2
  },
  {
    "lastName": "Aaron",
    "middleInitial": "G",
    "firstName": "Patricia",
    "jobClass": "FACILITIES/OFFICE SERVICES II",
    "agencyName": "Mayor's Office of Employment Development",
    "agencyID": "A03",
    "annualSalary": 63457,
    "grossPay": 29461.96,
    "hireDate": "1979-10-24",
    "fiscalYear": "FY2021",
    "ObjectId": 3
  },
  {
    "lastName": "Abadir",
    "middleInitial": "O",
    "firstName": "Adam",
    "jobClass": "OPERATIONS OFFICER II",
    "agencyName": "Health Department",
    "agencyID": "A65",
    "annualSalary": 74299,
    "grossPay": 85450.35,
    "hireDate": "2016-12-12",
    "fiscalYear": "FY2021",
    "ObjectId": 4
  },
  {
    "lastName": "Abaku",
    "middleInitial": "O",
    "firstName": "Aigbolosimuan",
    "jobClass": "POLICE OFFICER EID",
    "agencyName": "Police Department",
    "agencyID": "A99",
    "annualSalary": 68226,
    "grossPay": 102775.97,
    "hireDate": "2018-04-17",
    "fiscalYear": "FY2021",
    "ObjectId": 5
  },
  {
    "lastName": "Abbeduto",
    "middleInitial": "NA",
    "firstName": "Mack",
    "jobClass": "ASSISTANT STATE'S ATTORNEY",
    "agencyName": "State's Attorney's Office",
    "agencyID": "A29",
    "annualSalary": 72033,
    "grossPay": 73367.2,
    "hireDate": "2017-05-22",
    "fiscalYear": "FY2021",
    "ObjectId": 6
  },
  {
    "lastName": "Abbott-Cole",
    "middleInitial": "NA",
    "firstName": "Michelle",
    "jobClass": "OPERATIONS OFFICER III",
    "agencyName": "Transportation - Traffic",
    "agencyID": "A90",
    "annualSalary": 79827,
    "grossPay": 81305.63,
    "hireDate": "2014-11-28",
    "fiscalYear": "FY2021",
    "ObjectId": 7
  },
  {
    "lastName": "Abdal-Rahim",
    "middleInitial": "A",
    "firstName": "Naim",
    "jobClass": "FIRE PUMP OPERATOR SUPPRESSION",
    "agencyName": "Fire Department",
    "agencyID": "A64",
    "annualSalary": 72007,
    "grossPay": 103168.86,
    "hireDate": "2011-03-30",
    "fiscalYear": "FY2021",
    "ObjectId": 8
  },
  {
    "lastName": "Abdi",
    "middleInitial": "W",
    "firstName": "Ezekiel",
    "jobClass": "POLICE SERGEANT",
    "agencyName": "Police Department",
    "agencyID": "A99",
    "annualSalary": 97058,
    "grossPay": 143215.4,
    "hireDate": "2007-06-14",
    "fiscalYear": "FY2021",
    "ObjectId": 9
  },
  {
    "lastName": "Abdrabou",
    "middleInitial": "NA",
    "firstName": "Fouad",
    "jobClass": "POLICE OFFICER",
    "agencyName": "Police Department",
    "agencyID": "A99",
    "annualSalary": 55117,
    "grossPay": 41074.56,
    "hireDate": "2020-08-25",
    "fiscalYear": "FY2021",
    "ObjectId": 10
  },
  {
    "lastName": "Abdul",
    "middleInitial": "NA",
    "firstName": "Jalil",
    "jobClass": "ENGINEER I",
    "agencyName": "Public Works - Water & Waste Water",
    "agencyID": "A50",
    "annualSalary": 67771,
    "grossPay": 69809.82,
    "hireDate": "2017-07-17",
    "fiscalYear": "FY2021",
    "ObjectId": 11
  },
  {
    "lastName": "Abdul Adl",
    "middleInitial": "A",
    "firstName": "Attrice",
    "jobClass": "RADIO DISPATCHER SHERIFF",
    "agencyName": "Sheriff's Office",
    "agencyID": "A38",
    "annualSalary": 52613,
    "grossPay": 71252.88,
    "hireDate": "1999-09-02",
    "fiscalYear": "FY2021",
    "ObjectId": 12
  },
  {
    "lastName": "Abdul Saboor",
    "middleInitial": "N",
    "firstName": "Dana",
    "jobClass": "COURT SECRETARY I",
    "agencyName": "Courts - Circuit Court",
    "agencyID": "A31",
    "annualSalary": 60785,
    "grossPay": 61231.4,
    "hireDate": "1998-04-13",
    "fiscalYear": "FY2021",
    "ObjectId": 13
  },
  {
    "lastName": "Abdul-Hamid",
    "middleInitial": "NA",
    "firstName": "Umar",
    "jobClass": "OPERATIONS TECH APPRENTICE",
    "agencyName": "Public Works - Water & Waste Water",
    "agencyID": "A50",
    "annualSalary": 37822,
    "grossPay": 62279.75,
    "hireDate": "1995-01-17",
    "fiscalYear": "FY2021",
    "ObjectId": 14
  },
  {
    "lastName": "Abdul-Jabbar",
    "middleInitial": "A",
    "firstName": "Bushra",
    "jobClass": "SOCIAL SERVICE COORDINATOR",
    "agencyName": "Health Department",
    "agencyID": "A65",
    "annualSalary": 48743,
    "grossPay": 51387.86,
    "hireDate": "2008-04-14",
    "fiscalYear": "FY2021",
    "ObjectId": 15
  },
  {
    "lastName": "Abdul-Khaliq",
    "middleInitial": "NA",
    "firstName": "Amahl",
    "jobClass": "RECREATION LEADER II",
    "agencyName": "Recreation & Parks - Recreation",
    "agencyID": "A04",
    "annualSalary": 34879,
    "grossPay": 28189.07,
    "hireDate": "2019-06-06",
    "fiscalYear": "FY2021",
    "ObjectId": 16
  },
  {
    "lastName": "Abdul-Saboor",
    "middleInitial": "NA",
    "firstName": "Jamillah",
    "jobClass": "PRINTER LIBRARY",
    "agencyName": "Enoch Pratt Free Library",
    "agencyID": "A75",
    "annualSalary": 46840,
    "grossPay": 46302.18,
    "hireDate": "2009-07-27",
    "fiscalYear": "FY2021",
    "ObjectId": 17
  },
  {
    "lastName": "Abdullah",
    "middleInitial": "NA",
    "firstName": "Abdul",
    "jobClass": "RESEARCH ANALYST II",
    "agencyName": "Police Department",
    "agencyID": "A99",
    "annualSalary": 76500,
    "grossPay": 11538.5,
    "hireDate": "2021-04-26",
    "fiscalYear": "FY2021",
    "ObjectId": 18
  },
  {
    "lastName": "Abdullah",
    "middleInitial": "A",
    "firstName": "Beverly",
    "jobClass": "ADMINISTRATIVE COORDINATOR",
    "agencyName": "Housing & Community Development",
    "agencyID": "A06",
    "annualSalary": 54316,
    "grossPay": 55713.59,
    "hireDate": "1986-12-01",
    "fiscalYear": "FY2021",
    "ObjectId": 19
  },
  {
    "lastName": "Abdullahi",
    "middleInitial": "M",
    "firstName": "Sharon",
    "jobClass": "911 OPERATOR",
    "agencyName": "Fire Department",
    "agencyID": "A64",
    "annualSalary": 59173,
    "grossPay": 68386.91,
    "hireDate": "2004-06-10",
    "fiscalYear": "FY2021",
    "ObjectId": 20
  },
  {
    "lastName": "Abdullateef",
    "middleInitial": "NA",
    "firstName": "Muhammed",
    "jobClass": "OPERATIONS OFFICER V (CIVIL SERVICE)",
    "agencyName": "General Services",
    "agencyID": "A85",
    "annualSalary": 90144,
    "grossPay": 92572.21,
    "hireDate": "2019-05-09",
    "fiscalYear": "FY2021",
    "ObjectId": 21
  },
  {
    "lastName": "Abdulrahman",
    "middleInitial": "H",
    "firstName": "Mustafa",
    "jobClass": "POLICE OFFICER",
    "agencyName": "Police Department",
    "agencyID": "A99",
    "annualSalary": 56087,
    "grossPay": 70958.48,
    "hireDate": "2018-12-28",
    "fiscalYear": "FY2021",
    "ObjectId": 22
  },
  {
    "lastName": "Abdur-Rahman",
    "middleInitial": "NA",
    "firstName": "Diane",
    "jobClass": "FACILITIES/OFFICE SERVICES I",
    "agencyName": "Mayor's Office of Employment Development",
    "agencyID": "A03",
    "annualSalary": 26646,
    "grossPay": 27526.74,
    "hireDate": "2017-03-27",
    "fiscalYear": "FY2021",
    "ObjectId": 23
  },
  {
    "lastName": "Abebe",
    "middleInitial": "E",
    "firstName": "Miraf",
    "jobClass": "AUDITOR II",
    "agencyName": "Comptroller - Audits",
    "agencyID": "A24",
    "annualSalary": 71212,
    "grossPay": 72531.15,
    "hireDate": "2012-02-06",
    "fiscalYear": "FY2021",
    "ObjectId": 24
  },
  {
    "lastName": "Abekeh",
    "middleInitial": "NA",
    "firstName": "Michelle",
    "jobClass": "RESEARCH ANALYST I",
    "agencyName": "Health Department",
    "agencyID": "A65",
    "annualSalary": 46074,
    "grossPay": 12307.99,
    "hireDate": "2020-12-14",
    "fiscalYear": "FY2021",
    "ObjectId": 25
  },
  {
    "lastName": "Abel",
    "middleInitial": "E",
    "firstName": "Patrice",
    "jobClass": "CONTACT TRACER SUPPORT",
    "agencyName": "Health Department",
    "agencyID": "A65",
    "annualSalary": 35000,
    "grossPay": 30721.76,
    "hireDate": "2020-09-24",
    "fiscalYear": "FY2021",
    "ObjectId": 26
  },
  {
    "lastName": "Abend-Kollin",
    "middleInitial": "NA",
    "firstName": "Emily",
    "jobClass": "FLEET QUALITY CONTROL ANALYST",
    "agencyName": "General Services",
    "agencyID": "A85",
    "annualSalary": 57302,
    "grossPay": 67559.21,
    "hireDate": "2017-01-05",
    "fiscalYear": "FY2021",
    "ObjectId": 27
  },
  {
    "lastName": "Abera",
    "middleInitial": "M",
    "firstName": "Theodros",
    "jobClass": "CONTRACT SERVICES SPECIALIST II",
    "agencyName": "Baltimore City Office of Information and Technology",
    "agencyID": "A40",
    "annualSalary": 171600,
    "grossPay": 192175.24,
    "hireDate": "2020-03-02",
    "fiscalYear": "FY2021",
    "ObjectId": 28
  },
  {
    "lastName": "Abid",
    "middleInitial": "NA",
    "firstName": "Amal",
    "jobClass": "ENGINEER II",
    "agencyName": "Transportation - Highways",
    "agencyID": "A49",
    "annualSalary": 76282,
    "grossPay": 78281.85,
    "hireDate": "2013-12-02",
    "fiscalYear": "FY2021",
    "ObjectId": 29
  },
  {
    "lastName": "Abraham",
    "middleInitial": "D",
    "firstName": "Donta",
    "jobClass": "LABORER",
    "agencyName": "Public Works - Solid Waste (weekly)",
    "agencyID": "B70",
    "annualSalary": 39597,
    "grossPay": 48069.15,
    "hireDate": "2000-10-16",
    "fiscalYear": "FY2021",
    "ObjectId": 30
  },
  {
    "lastName": "Abraham",
    "middleInitial": "NA",
    "firstName": "Pinto",
    "jobClass": "POLICE OFFICER TRAINEE",
    "agencyName": "Police Department",
    "agencyID": "A99",
    "annualSalary": 55117,
    "grossPay": 31149.53,
    "hireDate": "2020-11-10",
    "fiscalYear": "FY2021",
    "ObjectId": 31
  },
  {
    "lastName": "Abraham",
    "middleInitial": "NA",
    "firstName": "Santhosh",
    "jobClass": "FISCAL SUPERVISOR",
    "agencyName": "Public Works - Water & Waste Water",
    "agencyID": "A50",
    "annualSalary": 70151,
    "grossPay": 71450.92,
    "hireDate": "2007-12-31",
    "fiscalYear": "FY2021",
    "ObjectId": 32
  },
  {
    "lastName": "Abraham",
    "middleInitial": "M",
    "firstName": "Sharon",
    "jobClass": "HOUSING INSPECTOR",
    "agencyName": "Housing & Community Development",
    "agencyID": "A06",
    "annualSalary": 58319,
    "grossPay": 66054.71,
    "hireDate": "1997-12-15",
    "fiscalYear": "FY2021",
    "ObjectId": 33
  },
  {
    "lastName": "Abraham",
    "middleInitial": "J",
    "firstName": "Terence",
    "jobClass": "CONSTRUCTION PROJECT SUPV II",
    "agencyName": "General Services",
    "agencyID": "A85",
    "annualSalary": 90586,
    "grossPay": 89114.48,
    "hireDate": "2015-10-29",
    "fiscalYear": "FY2021",
    "ObjectId": 34
  },
  {
    "lastName": "Abrahams",
    "middleInitial": "A",
    "firstName": "Brandon",
    "jobClass": "POLICE OFFICER",
    "agencyName": "Police Department",
    "agencyID": "A99",
    "annualSalary": 75167,
    "grossPay": 77063.99,
    "hireDate": "2015-01-20",
    "fiscalYear": "FY2021",
    "ObjectId": 35
  },
  {
    "lastName": "Abu-Hakim",
    "middleInitial": "NA",
    "firstName": "Kendall",
    "jobClass": "GEN SUPT TRANS MAINTENANCE",
    "agencyName": "Transportation - Highways",
    "agencyID": "A49",
    "annualSalary": 75310,
    "grossPay": 82948.65,
    "hireDate": "2012-01-09",
    "fiscalYear": "FY2021",
    "ObjectId": 36
  },
  {
    "lastName": "Abugo",
    "middleInitial": "NA",
    "firstName": "Susan",
    "jobClass": "COMMUNITY HLTH NURSE II 10MTH",
    "agencyName": "Health Department",
    "agencyID": "A65",
    "annualSalary": 67648,
    "grossPay": 75542.31,
    "hireDate": "2003-08-27",
    "fiscalYear": "FY2021",
    "ObjectId": 37
  },
  {
    "lastName": "Abukhdeir",
    "middleInitial": "M",
    "firstName": "Abrar",
    "jobClass": "OPERATIONS OFFICER V",
    "agencyName": "General Services",
    "agencyID": "A85",
    "annualSalary": 96912,
    "grossPay": 98453.4,
    "hireDate": "2008-06-21",
    "fiscalYear": "FY2021",
    "ObjectId": 38
  },
  {
    "lastName": "Abukhdeir",
    "middleInitial": "M",
    "firstName": "Tasnem",
    "jobClass": "PUBLIC HEALTH REP II",
    "agencyName": "Health Department",
    "agencyID": "A65",
    "annualSalary": 46291,
    "grossPay": 48460.1,
    "hireDate": "2017-07-13",
    "fiscalYear": "FY2021",
    "ObjectId": 39
  },
  {
    "lastName": "Achhammer",
    "middleInitial": "J",
    "firstName": "Matthew",
    "jobClass": "CONTRACT SERVICES SPECIALIST II",
    "agencyName": "Liquor License Board",
    "agencyID": "A09",
    "annualSalary": 81245,
    "grossPay": 67568.68,
    "hireDate": "2018-04-09",
    "fiscalYear": "FY2021",
    "ObjectId": 40
  },
  {
    "lastName": "Ackerman",
    "middleInitial": "M",
    "firstName": "Heidi",
    "jobClass": "ELECTION JUDGES REGULAR",
    "agencyName": "D01",
    "agencyID": "D01",
    "annualSalary": 0,
    "grossPay": 185,
    "hireDate": "2020-12-08",
    "fiscalYear": "FY2021",
    "ObjectId": 41
  },
  {
    "lastName": "Ackwood",
    "middleInitial": "NA",
    "firstName": "Karen",
    "jobClass": "SERVICE ASSISTANT LIBRARY",
    "agencyName": "Enoch Pratt Free Library",
    "agencyID": "A75",
    "annualSalary": 22880,
    "grossPay": 11183.35,
    "hireDate": "1992-02-19",
    "fiscalYear": "FY2021",
    "ObjectId": 42
  },
  {
    "lastName": "Ackwood",
    "middleInitial": "E",
    "firstName": "Kristina",
    "jobClass": "911 OPERATOR",
    "agencyName": "Fire Department",
    "agencyID": "A64",
    "annualSalary": 55823,
    "grossPay": 62404.47,
    "hireDate": "2016-09-12",
    "fiscalYear": "FY2021",
    "ObjectId": 43
  },
  {
    "lastName": "Acosta",
    "middleInitial": "NA",
    "firstName": "Alexis",
    "jobClass": "POLICE OFFICER",
    "agencyName": "Police Department",
    "agencyID": "A99",
    "annualSalary": 55117,
    "grossPay": 47157.04,
    "hireDate": "2020-08-25",
    "fiscalYear": "FY2021",
    "ObjectId": 44
  },
  {
    "lastName": "Acosta",
    "middleInitial": "NA",
    "firstName": "Arnulfo",
    "jobClass": "POLICE OFFICER",
    "agencyName": "Police Department",
    "agencyID": "A99",
    "annualSalary": 55249,
    "grossPay": 69143.95,
    "hireDate": "2019-07-08",
    "fiscalYear": "FY2021",
    "ObjectId": 45
  },
  {
    "lastName": "Acosta",
    "middleInitial": "J",
    "firstName": "Heath",
    "jobClass": "POLICE OFFICER",
    "agencyName": "Police Department",
    "agencyID": "A99",
    "annualSalary": 55117,
    "grossPay": 59308.76,
    "hireDate": "2020-05-21",
    "fiscalYear": "FY2021",
    "ObjectId": 46
  },
  {
    "lastName": "Acosta",
    "middleInitial": "A",
    "firstName": "Jahmoor",
    "jobClass": "POLICE OFFICER",
    "agencyName": "Police Department",
    "agencyID": "A99",
    "annualSalary": 70542,
    "grossPay": 83407.17,
    "hireDate": "2016-10-03",
    "fiscalYear": "FY2021",
    "ObjectId": 47
  },
  {
    "lastName": "Acosta Lopez",
    "middleInitial": "NA",
    "firstName": "Bryan",
    "jobClass": "POLICE OFFICER",
    "agencyName": "Police Department",
    "agencyID": "A99",
    "annualSalary": 70542,
    "grossPay": 79662.93,
    "hireDate": "2017-02-13",
    "fiscalYear": "FY2021",
    "ObjectId": 48
  },
  {
    "lastName": "Acquaviva",
    "middleInitial": "NA",
    "firstName": "Megan",
    "jobClass": "ASSISTANT STATE'S ATTORNEY",
    "agencyName": "State's Attorney's Office",
    "agencyID": "A29",
    "annualSalary": 78488,
    "grossPay": 79941.56,
    "hireDate": "2013-10-07",
    "fiscalYear": "FY2021",
    "ObjectId": 49
  },
  {
    "lastName": "Acree",
    "middleInitial": "D",
    "firstName": "Ennis",
    "jobClass": "FIRE PUMP OPERATOR SUPP ALS",
    "agencyName": "Fire Department",
    "agencyID": "A64",
    "annualSalary": 76441,
    "grossPay": 89605.29,
    "hireDate": "2002-10-07",
    "fiscalYear": "FY2021",
    "ObjectId": 50
  },
  {
    "lastName": "Amos",
    "middleInitial": "S",
    "firstName": "Darryl",
    "jobClass": "PUBLIC WORKS INSPECTOR II",
    "agencyName": "Public Works - Water & Waste Water",
    "agencyID": "A50",
    "annualSalary": 55109,
    "grossPay": 62424.71,
    "hireDate": "2008-01-14",
    "fiscalYear": "FY2021",
    "ObjectId": 51
  },
  {
    "lastName": "Amos",
    "middleInitial": "L",
    "firstName": "Virginia",
    "jobClass": "POLICE SERGEANT",
    "agencyName": "Police Department",
    "agencyID": "A99",
    "annualSalary": 110585,
    "grossPay": 125893.71,
    "hireDate": "1988-11-28",
    "fiscalYear": "FY2021",
    "ObjectId": 52
  },
  {
    "lastName": "Amponsah",
    "middleInitial": "NA",
    "firstName": "Richard",
    "jobClass": "CONTACT TRACER MANAGEMENT",
    "agencyName": "Health Department",
    "agencyID": "A65",
    "annualSalary": 70000,
    "grossPay": 51522.09,
    "hireDate": "2020-11-05",
    "fiscalYear": "FY2021",
    "ObjectId": 53
  },
  {
    "lastName": "Amprey",
    "middleInitial": "M",
    "firstName": "Andrea",
    "jobClass": "LICENSED GRAD SOCIAL WORKER (NON-CIVIL)",
    "agencyName": "State's Attorney's Office",
    "agencyID": "A29",
    "annualSalary": 66266,
    "grossPay": 68764.63,
    "hireDate": "2016-05-02",
    "fiscalYear": "FY2021",
    "ObjectId": 54
  },
  {
    "lastName": "Ampuero Cajan",
    "middleInitial": "M",
    "firstName": "Ana",
    "jobClass": "CHEMIST II",
    "agencyName": "Public Works - Water & Waste Water",
    "agencyID": "A50",
    "annualSalary": 58241,
    "grossPay": 66412.19,
    "hireDate": "2016-02-29",
    "fiscalYear": "FY2021",
    "ObjectId": 55
  },
  {
    "lastName": "Amsel",
    "middleInitial": "L",
    "firstName": "Christopher",
    "jobClass": "POLICE OFFICER",
    "agencyName": "Police Department",
    "agencyID": "A99",
    "annualSalary": 78080,
    "grossPay": 118444.41,
    "hireDate": "2014-02-12",
    "fiscalYear": "FY2021",
    "ObjectId": 56
  },
  {
    "lastName": "Amy",
    "middleInitial": "L",
    "firstName": "Kevin",
    "jobClass": "POLICE OFFICER",
    "agencyName": "Police Department",
    "agencyID": "A99",
    "annualSalary": 91355,
    "grossPay": 87260.52,
    "hireDate": "2000-10-26",
    "fiscalYear": "FY2021",
    "ObjectId": 57
  },
  {
    "lastName": "Anastasiades",
    "middleInitial": "NA",
    "firstName": "Demos",
    "jobClass": "BUILDING PROJECT COORDINATOR",
    "agencyName": "Recreation & Parks - Administration",
    "agencyID": "A67",
    "annualSalary": 59557,
    "grossPay": 64175.04,
    "hireDate": "2018-07-30",
    "fiscalYear": "FY2021",
    "ObjectId": 58
  },
  {
    "lastName": "Anbinder",
    "middleInitial": "D",
    "firstName": "Robert",
    "jobClass": "CHIEF SOLICITOR",
    "agencyName": "Law Department",
    "agencyID": "A30",
    "annualSalary": 120435,
    "grossPay": 123573.91,
    "hireDate": "1994-07-02",
    "fiscalYear": "FY2021",
    "ObjectId": 59
  },
  {
    "lastName": "Ancrum",
    "middleInitial": "K",
    "firstName": "Baron",
    "jobClass": "POLICE OFFICER",
    "agencyName": "Police Department",
    "agencyID": "A99",
    "annualSalary": 94534,
    "grossPay": 99159.9,
    "hireDate": "1996-09-09",
    "fiscalYear": "FY2021",
    "ObjectId": 60
  },
  {
    "lastName": "Andersen",
    "middleInitial": "H",
    "firstName": "William",
    "jobClass": "DESIGN PLANNER II",
    "agencyName": "Recreation & Parks - Administration",
    "agencyID": "A67",
    "annualSalary": 77467,
    "grossPay": 79795.88,
    "hireDate": "2009-04-27",
    "fiscalYear": "FY2021",
    "ObjectId": 61
  },
  {
    "lastName": "Anderson",
    "middleInitial": "NA",
    "firstName": "Andrea",
    "jobClass": "OMBUDSMAN",
    "agencyName": "Housing & Community Development",
    "agencyID": "A06",
    "annualSalary": 66731,
    "grossPay": 73623.54,
    "hireDate": "2014-10-27",
    "fiscalYear": "FY2021",
    "ObjectId": 62
  },
  {
    "lastName": "Anderson",
    "middleInitial": "NA",
    "firstName": "Anthony",
    "jobClass": "WATER SYSTEMS TREATMENT SUPV",
    "agencyName": "Public Works - Water & Waste Water",
    "agencyID": "A50",
    "annualSalary": 79120,
    "grossPay": 86825.19,
    "hireDate": "1978-07-24",
    "fiscalYear": "FY2021",
    "ObjectId": 63
  },
  {
    "lastName": "Anderson",
    "middleInitial": "NA",
    "firstName": "Antonio",
    "jobClass": "COMMUNITY EDUCATION AND ENGAGEMENT OFFICER",
    "agencyName": "MOCFS",
    "agencyID": "A10",
    "annualSalary": 40248,
    "grossPay": 21973.37,
    "hireDate": "2018-08-02",
    "fiscalYear": "FY2021",
    "ObjectId": 64
  },
  {
    "lastName": "Anderson",
    "middleInitial": "D",
    "firstName": "April",
    "jobClass": "PARAMEDIC NRP",
    "agencyName": "Fire Department",
    "agencyID": "A64",
    "annualSalary": 70698,
    "grossPay": 89753.77,
    "hireDate": "2017-07-12",
    "fiscalYear": "FY2021",
    "ObjectId": 65
  },
  {
    "lastName": "Anderson",
    "middleInitial": "NA",
    "firstName": "Cassidy",
    "jobClass": "EXECUTIVE ASSISTANT",
    "agencyName": "Mayor's Office",
    "agencyID": "A01",
    "annualSalary": 61200,
    "grossPay": 25384.77,
    "hireDate": "2021-01-25",
    "fiscalYear": "FY2021",
    "ObjectId": 66
  },
  {
    "lastName": "Anderson",
    "middleInitial": "NA",
    "firstName": "Catherine",
    "jobClass": "CROSSING GUARD",
    "agencyName": "Transportation - Crossing Guards",
    "agencyID": "C90",
    "annualSalary": 13259,
    "grossPay": 12680.85,
    "hireDate": "1990-06-02",
    "fiscalYear": "FY2021",
    "ObjectId": 67
  },
  {
    "lastName": "Anderson",
    "middleInitial": "NA",
    "firstName": "Charles",
    "jobClass": "POLICE SERGEANT EID",
    "agencyName": "Police Department",
    "agencyID": "A99",
    "annualSalary": 110179,
    "grossPay": 225093.62,
    "hireDate": "1997-09-30",
    "fiscalYear": "FY2021",
    "ObjectId": 68
  },
  {
    "lastName": "Anderson",
    "middleInitial": "O",
    "firstName": "Clemmie",
    "jobClass": "POLICE OFFICER",
    "agencyName": "Police Department",
    "agencyID": "A99",
    "annualSalary": 90562,
    "grossPay": 92835.95,
    "hireDate": "2003-03-31",
    "fiscalYear": "FY2021",
    "ObjectId": 69
  },
  {
    "lastName": "Anderson",
    "middleInitial": "L",
    "firstName": "Deidre",
    "jobClass": "PARALEGAL (CIVIL SERVICE)",
    "agencyName": "Housing & Community Development",
    "agencyID": "A06",
    "annualSalary": 64433,
    "grossPay": 67779.57,
    "hireDate": "2008-06-17",
    "fiscalYear": "FY2021",
    "ObjectId": 70
  },
  {
    "lastName": "Anderson",
    "middleInitial": "M",
    "firstName": "Demetrius",
    "jobClass": "LIFEGUARD I",
    "agencyName": "Recreation & Parks - Recreation",
    "agencyID": "A04",
    "annualSalary": 11.75,
    "grossPay": 440.63,
    "hireDate": "2021-05-29",
    "fiscalYear": "FY2021",
    "ObjectId": 71
  },
  {
    "lastName": "Anderson",
    "middleInitial": "A",
    "firstName": "Derrick",
    "jobClass": "SOLID WASTE WORKER",
    "agencyName": "Public Works - Solid Waste",
    "agencyID": "A70",
    "annualSalary": 35804,
    "grossPay": 45778.35,
    "hireDate": "2018-08-27",
    "fiscalYear": "FY2021",
    "ObjectId": 72
  },
  {
    "lastName": "Anderson",
    "middleInitial": "NA",
    "firstName": "Eugene",
    "jobClass": "LABORER",
    "agencyName": "Public Works - Solid Waste",
    "agencyID": "A70",
    "annualSalary": 22880,
    "grossPay": 44912.37,
    "hireDate": "2013-08-05",
    "fiscalYear": "FY2021",
    "ObjectId": 73
  },
  {
    "lastName": "Anderson",
    "middleInitial": "NA",
    "firstName": "Evan",
    "jobClass": "POLICE OFFICER",
    "agencyName": "Police Department",
    "agencyID": "A99",
    "annualSalary": 90562,
    "grossPay": 113570.72,
    "hireDate": "2002-05-06",
    "fiscalYear": "FY2021",
    "ObjectId": 74
  },
  {
    "lastName": "Anderson",
    "middleInitial": "B",
    "firstName": "Evelyn",
    "jobClass": "OFFICE SUPPORT SPECIALIST II",
    "agencyName": "Retire - Fire & Police",
    "agencyID": "A54",
    "annualSalary": 41412,
    "grossPay": 43289.22,
    "hireDate": "1971-07-29",
    "fiscalYear": "FY2021",
    "ObjectId": 75
  },
  {
    "lastName": "Anderson",
    "middleInitial": "A",
    "firstName": "Felicia",
    "jobClass": "CONTACT TRACER SUPPORT",
    "agencyName": "Health Department",
    "agencyID": "A65",
    "annualSalary": 35000,
    "grossPay": 31401.74,
    "hireDate": "2006-06-19",
    "fiscalYear": "FY2021",
    "ObjectId": 76
  },
  {
    "lastName": "Anderson",
    "middleInitial": "T",
    "firstName": "Garland",
    "jobClass": "PROCUREMENT SPECIALIST II",
    "agencyName": "Finance - Purchasing",
    "agencyID": "A17",
    "annualSalary": 89038,
    "grossPay": 88042.36,
    "hireDate": "2017-02-16",
    "fiscalYear": "FY2021",
    "ObjectId": 77
  },
  {
    "lastName": "Anderson",
    "middleInitial": "T",
    "firstName": "Garnetta",
    "jobClass": "RADIO DISPATCHER II",
    "agencyName": "Public Works - Water & Waste Water",
    "agencyID": "A50",
    "annualSalary": 45014,
    "grossPay": 50429.5,
    "hireDate": "2007-04-07",
    "fiscalYear": "FY2021",
    "ObjectId": 78
  },
  {
    "lastName": "Anderson",
    "middleInitial": "K",
    "firstName": "Ian",
    "jobClass": "SWIMMING POOL OPERATOR",
    "agencyName": "Recreation & Parks - Recreation",
    "agencyID": "A04",
    "annualSalary": 28554,
    "grossPay": 3826.07,
    "hireDate": "2011-06-08",
    "fiscalYear": "FY2021",
    "ObjectId": 79
  },
  {
    "lastName": "Anderson",
    "middleInitial": "NA",
    "firstName": "Jacqueline",
    "jobClass": "HEALTH PROJECT DIRECTOR",
    "agencyName": "Health Department",
    "agencyID": "A65",
    "annualSalary": 82399,
    "grossPay": 84242.2,
    "hireDate": "2016-08-09",
    "fiscalYear": "FY2021",
    "ObjectId": 80
  },
  {
    "lastName": "Anderson",
    "middleInitial": "W",
    "firstName": "Javon",
    "jobClass": "OPERATIONS CREW LEADER",
    "agencyName": "Convention Center",
    "agencyID": "A91",
    "annualSalary": 40279,
    "grossPay": 41267.16,
    "hireDate": "2007-06-23",
    "fiscalYear": "FY2021",
    "ObjectId": 81
  },
  {
    "lastName": "Anderson",
    "middleInitial": "L",
    "firstName": "Jennifer",
    "jobClass": "CRIME LABORATORY TECH SUPV",
    "agencyName": "Police Department",
    "agencyID": "A99",
    "annualSalary": 59824,
    "grossPay": 68502.89,
    "hireDate": "2014-08-07",
    "fiscalYear": "FY2021",
    "ObjectId": 82
  },
  {
    "lastName": "Anderson",
    "middleInitial": "W",
    "firstName": "John",
    "jobClass": "SHERIFF",
    "agencyName": "Sheriff's Office",
    "agencyID": "A38",
    "annualSalary": 157139,
    "grossPay": 164506.76,
    "hireDate": "1972-11-02",
    "fiscalYear": "FY2021",
    "ObjectId": 83
  },
  {
    "lastName": "Anderson",
    "middleInitial": "N",
    "firstName": "Joseph",
    "jobClass": "RADIO DISPATCHER II",
    "agencyName": "Public Works - Water & Waste Water",
    "agencyID": "A50",
    "annualSalary": 48947,
    "grossPay": 66483.54,
    "hireDate": "1991-08-05",
    "fiscalYear": "FY2021",
    "ObjectId": 84
  },
  {
    "lastName": "Anderson",
    "middleInitial": "NA",
    "firstName": "Justin",
    "jobClass": "SEASONAL MAINT AIDE",
    "agencyName": "Public Works - Solid Waste",
    "agencyID": "A70",
    "annualSalary": 22880,
    "grossPay": 16589.47,
    "hireDate": "2020-09-07",
    "fiscalYear": "FY2021",
    "ObjectId": 85
  },
  {
    "lastName": "Anderson",
    "middleInitial": "E",
    "firstName": "Karl",
    "jobClass": "LABORER",
    "agencyName": "Public Works - Water & Waste Water",
    "agencyID": "A50",
    "annualSalary": 36327,
    "grossPay": 52406.65,
    "hireDate": "2013-12-09",
    "fiscalYear": "FY2021",
    "ObjectId": 86
  },
  {
    "lastName": "Anderson",
    "middleInitial": "A",
    "firstName": "Kenneth",
    "jobClass": "CDL DRIVER II",
    "agencyName": "Transportation - Highways",
    "agencyID": "A49",
    "annualSalary": 50293,
    "grossPay": 83474.72,
    "hireDate": "2006-07-31",
    "fiscalYear": "FY2021",
    "ObjectId": 87
  },
  {
    "lastName": "Anderson",
    "middleInitial": "A",
    "firstName": "Kenneth",
    "jobClass": "CDL DRIVER II",
    "agencyName": "Transportation - Highways",
    "agencyID": "A49",
    "annualSalary": 50293,
    "grossPay": 83474.72,
    "hireDate": "2006-07-31",
    "fiscalYear": "FY2021",
    "ObjectId": 88
  },
  {
    "lastName": "Anderson",
    "middleInitial": "A",
    "firstName": "Kenneth",
    "jobClass": "CDL DRIVER II",
    "agencyName": "Transportation - Highways",
    "agencyID": "A49",
    "annualSalary": 42455,
    "grossPay": 87471.24,
    "hireDate": "2014-10-14",
    "fiscalYear": "FY2021",
    "ObjectId": 89
  },
  {
    "lastName": "Anderson",
    "middleInitial": "A",
    "firstName": "Kenneth",
    "jobClass": "CDL DRIVER II",
    "agencyName": "Transportation - Highways",
    "agencyID": "A49",
    "annualSalary": 42455,
    "grossPay": 87471.24,
    "hireDate": "2014-10-14",
    "fiscalYear": "FY2021",
    "ObjectId": 90
  },
  {
    "lastName": "Anderson",
    "middleInitial": "J",
    "firstName": "Kevin",
    "jobClass": "LABORER",
    "agencyName": "Public Works - Solid Waste (weekly)",
    "agencyID": "B70",
    "annualSalary": 37417,
    "grossPay": 43885.33,
    "hireDate": "2006-09-25",
    "fiscalYear": "FY2021",
    "ObjectId": 91
  },
  {
    "lastName": "Anderson",
    "middleInitial": "K",
    "firstName": "Kirsten",
    "jobClass": "SOLID WASTE SUPERVISOR",
    "agencyName": "Public Works - Solid Waste",
    "agencyID": "A70",
    "annualSalary": 59988,
    "grossPay": 96797.73,
    "hireDate": "2006-07-05",
    "fiscalYear": "FY2021",
    "ObjectId": 92
  },
  {
    "lastName": "Anderson",
    "middleInitial": "NA",
    "firstName": "Lafawn",
    "jobClass": "SOLID WASTE DRIVER",
    "agencyName": "Public Works - Solid Waste",
    "agencyID": "A70",
    "annualSalary": 42607,
    "grossPay": 55972.58,
    "hireDate": "2015-08-10",
    "fiscalYear": "FY2021",
    "ObjectId": 93
  },
  {
    "lastName": "Anderson",
    "middleInitial": "K",
    "firstName": "Latonya",
    "jobClass": "911 OPERATOR",
    "agencyName": "Fire Department",
    "agencyID": "A64",
    "annualSalary": 55823,
    "grossPay": 90378.02,
    "hireDate": "2016-09-13",
    "fiscalYear": "FY2021",
    "ObjectId": 94
  },
  {
    "lastName": "Anderson",
    "middleInitial": "R",
    "firstName": "Lawrence",
    "jobClass": "OPERATIONS MANAGER II",
    "agencyName": "City Council",
    "agencyID": "A02",
    "annualSalary": 142500,
    "grossPay": 85900.48,
    "hireDate": "2020-11-16",
    "fiscalYear": "FY2021",
    "ObjectId": 95
  },
  {
    "lastName": "Anderson",
    "middleInitial": "A",
    "firstName": "Michael",
    "jobClass": "SOLID WASTE SUPERVISOR",
    "agencyName": "Public Works - Solid Waste",
    "agencyID": "A70",
    "annualSalary": 48440,
    "grossPay": 75042.51,
    "hireDate": "2006-12-18",
    "fiscalYear": "FY2021",
    "ObjectId": 96
  },
  {
    "lastName": "Anderson",
    "middleInitial": "C",
    "firstName": "Nicole",
    "jobClass": "OPERATIONS TECH APPRENTICE",
    "agencyName": "Public Works - Water & Waste Water",
    "agencyID": "A50",
    "annualSalary": 42086,
    "grossPay": 67268.02,
    "hireDate": "2001-02-05",
    "fiscalYear": "FY2021",
    "ObjectId": 97
  },
  {
    "lastName": "Anderson",
    "middleInitial": "M",
    "firstName": "Nicole",
    "jobClass": "COMMUNITY AIDE",
    "agencyName": "Transportation - Highways",
    "agencyID": "A49",
    "annualSalary": 27040,
    "grossPay": 25626.57,
    "hireDate": "2019-12-02",
    "fiscalYear": "FY2021",
    "ObjectId": 98
  },
  {
    "lastName": "Anderson",
    "middleInitial": "NA",
    "firstName": "Patrice",
    "jobClass": "TEMP SUMMER EMPL MOMR",
    "agencyName": "Mayor's Office of Employment Development",
    "agencyID": "A03",
    "annualSalary": 21.4,
    "grossPay": 1270,
    "hireDate": "2021-06-14",
    "fiscalYear": "FY2021",
    "ObjectId": 99
  },
  {
    "lastName": "Anderson",
    "middleInitial": "L",
    "firstName": "Philip",
    "jobClass": "ANALYST/PROGRAMMER II",
    "agencyName": "Public Works - Water & Waste Water",
    "agencyID": "A50",
    "annualSalary": 76206,
    "grossPay": 89847.7,
    "hireDate": "1988-11-14",
    "fiscalYear": "FY2021",
    "ObjectId": 100
  },
  {
    "lastName": "Adair",
    "middleInitial": "E",
    "firstName": "Raimond",
    "jobClass": "TRAFFIC SIGNAL INSTALLER I",
    "agencyName": "Transportation - Highways",
    "agencyID": "A49",
    "annualSalary": 37559,
    "grossPay": 47413.25,
    "hireDate": "2006-06-24",
    "fiscalYear": "FY2021",
    "ObjectId": 101
  },
  {
    "lastName": "Adam",
    "middleInitial": "NA",
    "firstName": "Idrissou",
    "jobClass": "CONTACT TRACER SUPPORT",
    "agencyName": "Health Department",
    "agencyID": "A65",
    "annualSalary": 35000,
    "grossPay": 16555.74,
    "hireDate": "2021-01-28",
    "fiscalYear": "FY2021",
    "ObjectId": 102
  },
  {
    "lastName": "Adam",
    "middleInitial": "NA",
    "firstName": "Maiga",
    "jobClass": "LIFEGUARD I",
    "agencyName": "Recreation & Parks - Recreation",
    "agencyID": "A04",
    "annualSalary": 23920,
    "grossPay": 4832.63,
    "hireDate": "2019-07-01",
    "fiscalYear": "FY2021",
    "ObjectId": 103
  },
  {
    "lastName": "Adamo",
    "middleInitial": "NA",
    "firstName": "Janice",
    "jobClass": "EXECUTIVE ASSISTANT",
    "agencyName": "Mayor's Office",
    "agencyID": "A01",
    "annualSalary": 55000,
    "grossPay": 47894.92,
    "hireDate": "2020-08-10",
    "fiscalYear": "FY2021",
    "ObjectId": 104
  },
  {
    "lastName": "Adams",
    "middleInitial": "S",
    "firstName": "Andrew",
    "jobClass": "OFFICE ASSISTANT II",
    "agencyName": "Enoch Pratt Free Library",
    "agencyID": "A75",
    "annualSalary": 34849,
    "grossPay": 33174.41,
    "hireDate": "2017-01-03",
    "fiscalYear": "FY2021",
    "ObjectId": 105
  },
  {
    "lastName": "Adams",
    "middleInitial": "K",
    "firstName": "Blair",
    "jobClass": "FIRE PRESS OFFICER",
    "agencyName": "Fire Department",
    "agencyID": "A64",
    "annualSalary": 96190,
    "grossPay": 101606.71,
    "hireDate": "2014-04-14",
    "fiscalYear": "FY2021",
    "ObjectId": 106
  },
  {
    "lastName": "Adams",
    "middleInitial": "L",
    "firstName": "Brandon",
    "jobClass": "UTILITY METER TECH II DPW",
    "agencyName": "Public Works - Water & Waste Water",
    "agencyID": "A50",
    "annualSalary": 42068,
    "grossPay": 44926.04,
    "hireDate": "2008-03-17",
    "fiscalYear": "FY2021",
    "ObjectId": 107
  },
  {
    "lastName": "Adams",
    "middleInitial": "L",
    "firstName": "Carla",
    "jobClass": "HR GENERALIST II",
    "agencyName": "Public Works - Administration",
    "agencyID": "A41",
    "annualSalary": 73450,
    "grossPay": 75110.25,
    "hireDate": "1995-01-30",
    "fiscalYear": "FY2021",
    "ObjectId": 108
  },
  {
    "lastName": "Adams",
    "middleInitial": "R",
    "firstName": "Damon",
    "jobClass": "POLICE OFFICER",
    "agencyName": "Police Department",
    "agencyID": "A99",
    "annualSalary": 88971,
    "grossPay": 120351.99,
    "hireDate": "2003-10-23",
    "fiscalYear": "FY2021",
    "ObjectId": 109
  },
  {
    "lastName": "Adams",
    "middleInitial": "M",
    "firstName": "Darryl",
    "jobClass": "PC SUPPORT TECHNICIAN II",
    "agencyName": "Public Works - Administration",
    "agencyID": "A41",
    "annualSalary": 46291,
    "grossPay": 52505.74,
    "hireDate": "2019-10-21",
    "fiscalYear": "FY2021",
    "ObjectId": 110
  },
  {
    "lastName": "Adams",
    "middleInitial": "E",
    "firstName": "Deborah",
    "jobClass": "OFFICE SUPPORT SPECIALIST III",
    "agencyName": "Finance - Purchasing",
    "agencyID": "A17",
    "annualSalary": 39293,
    "grossPay": 39757.45,
    "hireDate": "2008-06-19",
    "fiscalYear": "FY2021",
    "ObjectId": 111
  },
  {
    "lastName": "Adams",
    "middleInitial": "J",
    "firstName": "Douglas",
    "jobClass": "UTILITY METER TECH II DPW",
    "agencyName": "Public Works - Water & Waste Water",
    "agencyID": "A50",
    "annualSalary": 44518,
    "grossPay": 45755.02,
    "hireDate": "2000-12-26",
    "fiscalYear": "FY2021",
    "ObjectId": 112
  },
  {
    "lastName": "Adams",
    "middleInitial": "O",
    "firstName": "James",
    "jobClass": "PUBLIC WORKS INSPECTOR I",
    "agencyName": "Public Works - Water & Waste Water",
    "agencyID": "A50",
    "annualSalary": 46566,
    "grossPay": 58322.97,
    "hireDate": "2001-06-11",
    "fiscalYear": "FY2021",
    "ObjectId": 113
  },
  {
    "lastName": "Adams",
    "middleInitial": "T",
    "firstName": "Jermaine",
    "jobClass": "ADMINISTRATIVE COORDINATOR",
    "agencyName": "Human Resources",
    "agencyID": "A83",
    "annualSalary": 44263,
    "grossPay": 48030.29,
    "hireDate": "2019-07-13",
    "fiscalYear": "FY2021",
    "ObjectId": 114
  }
]
# Insert data into the index
# index.upsert([
#     ("A", [0.1]),
#     ("B", [0.2]),
#     ("C", [0.3]),
#     ("D", [0.4]),
#     ("E", [0.5])
# ])
for entry in json_data:
  index.upsert([
    (entry['agencyID'],[entry['annualSalary']]),
    ])
  # print(type(entry['annualSalary']))

# Describe index statistics


# Define a different query vector
data = index.describe_index_stats()
# print(data)
# Perform the query
getvalue=input("Enter the value to which you want salaries to be shown closest.")
query_response=index.query(
  vector=[getvalue], 
  top_k=10,
  include_values=True
)
# Print query response
print(query_response)
for matches in query_response.matches:
    print(matches.score)

# List active indexes
active_indexes = pinecone.list_indexes()
print(active_indexes)
