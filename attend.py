from datetime import datetime
def mark_attendance(name):
    with open ('attendancelist.csv','r+') as f:
        mydatalist=f.readlines()
        namelist=[]
        for line in mydatalist:
            entry=line.split(',') 
            namelist.append(entry[0])
        if name not in namelist:
            now=datetime.now()
            datestring=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{datestring} ')
mark_attendance('elon')