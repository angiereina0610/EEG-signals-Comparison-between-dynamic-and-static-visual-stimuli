import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyedflib
import os
#emotion mapping
dicMap = {
    'score_1':'Joy',  
    'score_2': 'Inspiration',  
    'score_3': 'Tenderness',    
    'score_4': 'Sadness',    
    'score_5': 'Fear',  
    'score_6': 'Disgust',  
    'score_7': 'Arousal',  
    'score_8': 'Valence',  
    'score_9': 'Familiarity',  
    'score_10': 'Liking'

}
#emotion labels by the authors
allOrder = [
['joy5', 'ins5', 'joy8', 'fear8', 'sad8', 'dis5', 'neu4', 'neu5', 'neu8', 'ten5', 'ten8', 'joy4', 'dis4', 'fear4', 'sad4', 'ins8', 'ins4', 'ten4', 'dis8', 'fear5', 'sad5'],
['fear8', 'fear5', 'dis4', 'ins8', 'joy8', 'ins4', 'neu4', 'neu8', 'neu5', 'sad4', 'dis8', 'fear4', 'ten5', 'ten8', 'joy4', 'dis5', 'sad8', 'sad5', 'joy5', 'ten4', 'ins5'],
['ten4', 'joy4', 'joy8', 'neu4', 'neu8', 'neu5', 'dis5', 'fear4', 'fear5', 'ten8', 'ten5', 'ins5', 'fear8', 'dis4', 'dis8', 'ins8', 'joy5', 'ins4', 'sad4', 'sad5', 'sad8'],
['fear5', 'dis8', 'dis5', 'joy4', 'ten5', 'ins5', 'neu4', 'neu8', 'neu5', 'sad8', 'fear8', 'sad4', 'ins4', 'ins8', 'joy8', 'fear4', 'sad5', 'dis4', 'ten4', 'joy5', 'ten8'],
['joy8', 'ten4', 'ins5', 'fear5', 'sad5', 'dis4', 'neu4', 'neu5', 'neu8', 'joy4', 'ten8', 'joy5', 'sad4', 'dis8', 'fear8', 'ins4', 'ten5', 'ins8', 'sad8', 'dis5', 'fear4'],
['joy8', 'ins5', 'ins8', 'dis4', 'dis8', 'fear8', 'ten4', 'joy5', 'ten5', 'dis5', 'fear5', 'fear4', 'ten8', 'ins4', 'joy4', 'sad8', 'sad4', 'sad5', 'neu4', 'neu5', 'neu8'],
['joy8', 'ten8', 'joy4', 'fear4', 'sad5', 'dis5', 'ins5', 'ten5', 'ten4', 'dis4', 'sad8', 'dis8', 'ins4', 'ins8', 'joy5', 'sad4', 'fear8', 'fear5', 'neu4', 'neu5', 'neu8'],
['neu4', 'neu5', 'neu8', 'dis8', 'sad4', 'fear5', 'ins4', 'ins5', 'ten5', 'dis4', 'sad8', 'fear4', 'ins8', 'joy4', 'ten8', 'fear8', 'dis5', 'sad5', 'ten4', 'joy8', 'joy5'],
['sad5', 'fear4', 'fear8', 'joy4', 'joy8', 'ten5', 'dis8', 'dis5', 'sad4', 'neu4', 'neu8', 'neu5', 'ins8', 'ten8', 'ins4', 'sad8', 'fear5', 'dis4', 'joy5', 'ten4', 'ins5'],
['sad4', 'fear5', 'sad8', 'joy8', 'ten8', 'joy4', 'sad5', 'dis8', 'fear4', 'neu4', 'neu8', 'neu5', 'ten4', 'ten5', 'ins4', 'dis4', 'fear8', 'dis5', 'joy5', 'ins5', 'ins8'],
['joy4', 'ins4', 'joy5', 'fear8', 'dis8', 'sad4', 'ten8', 'ins5', 'ten5', 'sad5', 'sad8', 'fear5', 'ins8', 'ten4', 'joy8', 'neu8', 'neu4', 'neu5', 'fear4', 'dis4', 'dis5'],
['sad8', 'fear5', 'fear8', 'ten8', 'ten5', 'joy8', 'fear4', 'sad4', 'sad5', 'neu4', 'neu8', 'neu5', 'ins8', 'ins4', 'ten4', 'dis5', 'dis8', 'dis4', 'joy5', 'joy4', 'ins5'],
['sad8', 'dis8', 'sad4', 'ten4', 'ten8', 'ins4', 'dis5', 'fear8', 'sad5', 'ten5', 'ins5', 'joy8', 'neu4', 'neu8', 'neu5', 'fear5', 'fear4', 'dis4', 'joy4', 'joy5', 'ins8'],
['ins8', 'ten4', 'ins5', 'neu4', 'neu8', 'neu5', 'sad5', 'dis4', 'sad4', 'ins4', 'ten8', 'ten5', 'dis8', 'sad8', 'fear8', 'joy5', 'joy4', 'joy8', 'fear4', 'fear5', 'dis5'],
['ins8', 'ten5', 'ten8', 'sad8', 'sad4', 'sad5', 'joy4', 'ins4', 'ins5', 'fear8', 'fear5', 'fear4', 'ten4', 'joy5', 'joy8', 'neu5', 'neu4', 'neu8', 'dis4', 'dis5', 'dis8'],
['fear4', 'dis4', 'fear8', 'ins8', 'joy8', 'ten8', 'dis5', 'sad4', 'dis8', 'ins5', 'ins4', 'joy4', 'neu8', 'neu4', 'neu5', 'fear5', 'sad8', 'sad5', 'joy5', 'ten5', 'ten4'],
['ten5', 'ins4', 'ins8', 'dis8', 'fear4', 'sad5', 'ins5', 'joy8', 'ten4', 'sad8', 'fear8', 'fear5', 'ten8', 'joy5', 'joy4', 'sad4', 'dis5', 'dis4', 'neu5', 'neu4', 'neu8'],
['neu4', 'neu5', 'neu8', 'sad4', 'dis8', 'dis5', 'joy4', 'ten4', 'ten5', 'sad5', 'fear5', 'fear4', 'ins5', 'ins4', 'ten8', 'dis4', 'fear8', 'sad8', 'joy8', 'ins8', 'joy5'],
['joy5', 'ten8', 'ins4', 'fear4', 'dis8', 'sad4', 'ten5', 'joy8', 'joy4', 'sad8', 'dis5', 'fear8', 'neu8', 'neu4', 'neu5', 'ins5', 'ten4', 'ins8', 'fear5', 'dis4', 'sad5'],
['joy5', 'ins8', 'joy4', 'neu4', 'neu5', 'neu8', 'fear4', 'sad4', 'fear8', 'ins5', 'ten4', 'ten5', 'dis4', 'sad8', 'sad5', 'ten8', 'ins4', 'joy8', 'dis5', 'fear5', 'dis8'],
['ten8', 'joy4', 'ins5', 'sad4', 'dis4', 'fear8', 'ins8', 'joy8', 'ins4', 'neu8', 'neu4', 'neu5', 'sad5', 'sad8', 'fear5', 'ten5', 'joy5', 'ten4', 'fear4', 'dis5', 'dis8'],
['joy5', 'ten8', 'ten4', 'dis4', 'fear4', 'fear5', 'joy8', 'ten5', 'joy4', 'sad5', 'sad8', 'dis8', 'neu5', 'neu8', 'neu4', 'ins4', 'ins5', 'ins8', 'fear8', 'sad4', 'dis5'],
['neu4', 'neu5', 'neu8', 'dis4', 'fear4', 'sad8', 'ins8', 'joy4', 'ten8', 'fear8', 'fear5', 'sad5', 'ten4', 'ins5', 'joy8', 'dis8', 'sad4', 'dis5', 'ten5', 'joy5', 'ins4'],
['joy5', 'ten5', 'ins4', 'fear4', 'sad8', 'sad4', 'ins5', 'ten4', 'ten8', 'sad5', 'fear5', 'fear8', 'ins8', 'joy8', 'joy4', 'dis8', 'dis5', 'dis4', 'neu8', 'neu4', 'neu5'],
['dis8', 'dis5', 'sad4', 'ins8', 'ten4', 'joy8', 'sad8', 'fear4', 'fear8', 'joy5', 'ins4', 'ten8', 'dis4', 'fear5', 'sad5', 'neu8', 'neu5', 'neu4', 'joy4', 'ins5', 'ten5'],
['fear4', 'sad5', 'fear8', 'ten4', 'ins5', 'joy8', 'dis4', 'dis8', 'sad8', 'ins4', 'joy5', 'joy4', 'dis5', 'sad4', 'fear5', 'ins8', 'ten8', 'ten5', 'neu5', 'neu8', 'neu4'],
['dis4', 'dis5', 'fear4', 'ins8', 'ins4', 'joy5', 'sad8', 'fear8', 'sad5', 'ins5', 'joy4', 'ten8', 'neu4', 'neu8', 'neu5', 'fear5', 'sad4', 'dis8', 'ten4', 'ten5', 'joy8'],
['ten4', 'ins5', 'joy4', 'dis5', 'sad5', 'fear4', 'ins8', 'joy8', 'ins4', 'fear5', 'fear8', 'dis8', 'neu5', 'neu8', 'neu4', 'ten8', 'joy5', 'ten5', 'sad4', 'dis4', 'sad8'],
['joy5', 'ten5', 'ins5', 'neu8', 'neu4', 'neu5', 'fear5', 'sad8', 'sad5', 'joy8', 'ten8', 'joy4', 'fear8', 'fear4', 'dis4', 'ten4', 'ins8', 'ins4', 'dis8', 'dis5', 'sad4'],
['sad8', 'dis8', 'dis5', 'joy5', 'ten4', 'joy4', 'sad5', 'fear5', 'fear8', 'ten8', 'ins8', 'ins4', 'sad4', 'fear4', 'dis4', 'joy8', 'ins5', 'ten5', 'neu5', 'neu8', 'neu4'],
['dis4', 'dis8', 'sad4', 'neu5', 'neu4', 'neu8', 'joy5', 'ins8', 'ins4', 'fear4', 'fear8', 'sad8', 'ins5', 'ten8', 'joy4', 'sad5', 'dis5', 'fear5', 'ten4', 'joy8', 'ten5'],
['joy5', 'joy4', 'ten4', 'sad5', 'fear5', 'fear4', 'ins5', 'ten8', 'ins8', 'dis8', 'dis5', 'sad8', 'ten5', 'ins4', 'joy8', 'sad4', 'fear8', 'dis4', 'neu5', 'neu8', 'neu4'],
['sad5', 'dis8', 'dis5', 'ins5', 'ten5', 'ten4', 'dis4', 'fear4', 'fear5', 'ten8', 'ins8', 'joy4', 'neu5', 'neu4', 'neu8', 'fear8', 'sad4', 'sad8', 'joy5', 'joy8', 'ins4'],
['ten5', 'ins5', 'joy4', 'sad4', 'fear5', 'fear4', 'ten8', 'joy8', 'ins8', 'dis8', 'sad5', 'dis5', 'joy5', 'ten4', 'ins4', 'dis4', 'fear8', 'sad8', 'neu4', 'neu8', 'neu5'],
['sad4', 'fear8', 'dis4', 'ins4', 'ins8', 'joy4', 'neu8', 'neu5', 'neu4', 'sad8', 'fear4', 'dis5', 'ten4', 'ten5', 'ten8', 'sad5', 'dis8', 'fear5', 'joy8', 'ins5', 'joy5'],
['joy5', 'joy4', 'joy8', 'dis4', 'dis8', 'fear5', 'neu5', 'neu8', 'neu4', 'ins4', 'ten5', 'ten4', 'dis5', 'sad5', 'fear4', 'ten8', 'ins8', 'ins5', 'sad4', 'sad8', 'fear8'],
['fear4', 'dis5', 'sad5', 'neu5', 'neu4', 'neu8', 'ins8', 'joy8', 'ten5', 'fear5', 'sad4', 'fear8', 'ins4', 'joy4', 'ten8', 'dis4', 'dis8', 'sad8', 'joy5', 'ins5', 'ten4'],
['joy8', 'ten8', 'ins8', 'fear8', 'sad4', 'fear5', 'ten4', 'ten5', 'joy5', 'sad8', 'dis4', 'fear4', 'neu4', 'neu5', 'neu8', 'ins5', 'ins4', 'joy4', 'sad5', 'dis8', 'dis5'],
['ins4', 'ten8', 'joy4', 'neu5', 'neu8', 'neu4', 'dis8', 'fear4', 'sad8', 'ins5', 'joy8', 'ten4', 'dis5', 'dis4', 'fear5', 'ins8', 'ten5', 'joy5', 'fear8', 'sad5', 'sad4'],
['ins4', 'ten4', 'ins5', 'sad5', 'dis5', 'fear4', 'neu5', 'neu8', 'neu4', 'ten5', 'ins8', 'joy4', 'sad8', 'fear5', 'sad4', 'ten8', 'joy5', 'joy8', 'dis8', 'dis4', 'fear8'],
['ins5', 'ten8', 'ins4', 'dis8', 'sad4', 'dis5', 'joy8', 'ten5', 'ins8', 'neu8', 'neu4', 'neu5', 'fear8', 'dis4', 'fear5', 'joy4', 'joy5', 'ten4', 'sad5', 'sad8', 'fear4'],
['ten8', 'ten4', 'joy8', 'dis8', 'sad5', 'sad4', 'joy5', 'ins8', 'ins4', 'neu4', 'neu5', 'neu8', 'fear4', 'dis4', 'fear5', 'ins5', 'ten5', 'joy4', 'dis5', 'fear8', 'sad8'],
['ins5', 'ten5', 'ins4', 'neu5', 'neu8', 'neu4', 'sad4', 'dis4', 'sad5', 'ins8', 'joy8', 'joy4', 'fear8', 'fear4', 'dis8', 'ten8', 'ten4', 'joy5', 'dis5', 'sad8', 'fear5'],
['sad8', 'dis5', 'dis4', 'joy5', 'ins5', 'joy8', 'sad5', 'sad4', 'fear5', 'ten4', 'ten8', 'ins4', 'neu8', 'neu5', 'neu4', 'dis8', 'fear8', 'fear4', 'joy4', 'ten5', 'ins8'],
['ins5', 'joy8', 'ins8', 'fear8', 'fear5', 'sad5', 'joy5', 'ten8', 'ten5', 'neu5', 'neu4', 'neu8', 'dis5', 'dis8', 'sad4', 'ins4', 'ten4', 'joy4', 'sad8', 'dis4', 'fear4'],
['fear5', 'dis5', 'dis8', 'ins5', 'ten5', 'ten8', 'neu8', 'neu4', 'neu5', 'fear8', 'dis4', 'sad4', 'ten4', 'ins8', 'ins4', 'sad5', 'sad8', 'fear4', 'joy8', 'joy4', 'joy5'],
['ins4', 'joy5', 'joy8', 'sad5', 'fear5', 'dis8', 'neu8', 'neu4', 'neu5', 'ins8', 'ten4', 'joy4', 'fear8', 'dis5', 'sad8', 'ins5', 'ten8', 'ten5', 'sad4', 'dis4', 'fear4'],
['joy5', 'ins8', 'ins5', 'dis8', 'dis5', 'fear5', 'ten4', 'ins4', 'joy8', 'dis4', 'fear4', 'sad5', 'ten8', 'ten5', 'joy4', 'fear8', 'sad8', 'sad4', 'neu8', 'neu4', 'neu5'],
['dis4', 'sad5', 'sad4', 'neu4', 'neu8', 'neu5', 'joy4', 'ten5', 'ten8', 'dis8', 'fear8', 'dis5', 'ins4', 'joy8', 'ten4', 'fear4', 'sad8', 'fear5', 'ins8', 'ins5', 'joy5'],
['ten5', 'ins8', 'ins4', 'neu4', 'neu8', 'neu5', 'fear4', 'fear8', 'dis4', 'joy4', 'ten4', 'ins5', 'fear5', 'sad5', 'dis8', 'ten8', 'joy8', 'joy5', 'sad4', 'sad8', 'dis5'],
['ten8', 'joy8', 'ten5', 'dis8', 'fear5', 'dis4', 'joy5', 'ten4', 'ins4', 'fear4', 'sad4', 'dis5', 'neu8', 'neu4', 'neu5', 'ins8', 'ins5', 'joy4', 'sad5', 'fear8', 'sad8'],
['joy4', 'joy5', 'ins8', 'fear5', 'dis5', 'dis8', 'neu5', 'neu4', 'neu8', 'joy8', 'ins5', 'ten5', 'sad5', 'fear4', 'dis4', 'ten4', 'ten8', 'ins4', 'sad8', 'fear8', 'sad4'],
['neu8', 'neu4', 'neu5', 'dis5', 'sad4', 'fear4', 'joy5', 'ins4', 'ten4', 'fear8', 'sad5', 'sad8', 'ten8', 'joy4', 'ten5', 'fear5', 'dis4', 'dis8', 'joy8', 'ins5', 'ins8']

]

#with the exception of one subject, since the labels are different
sub54Ima = ['ten8', 'ten5', 'ten4', 'fear8', 'fear5', 'fear4', 'dis8', 'dis5', 'dis4', 'joy8', 'joy5', 'joy4', 'sad8', 'sad5', 'sad4', 'neu8', 'neu5', 'neu4', 'ins8', 'ins5', 'ins4']
sub54Vid =  ['joy8', 'joy5', 'joy4', 'sad8', 'sad5', 'sad4', 'dis8', 'dis5', 'dis4', 'ins8', 'ins5', 'ins4', 'fear8', 'fear5', 'fear4', 'neu8', 'neu5', 'neu4', 'ten8', 'ten5', 'ten4']


reorder = ['sad4', 'sad5', 'sad8', 'dis4', 'dis5', 'dis8', 'fear4', 'fear5', 'fear8', 'neu4', 'neu5', 'neu8', 'joy4', 'joy5', 'joy8', 'ten4', 'ten5', 'ten8', 'ins4', 'ins5', 'ins8']



def pairingList( reoderList, subjectList ):

    orderArray = np.empty( len(subjectList) ) 
    idxEle = 0 

    for element in reoderList:
        
        idx = [ pos for pos in range(len(subjectList) ) if subjectList[pos] == element ]

        if len(idx)>1:
            print('error')

        orderArray[ idx[0]  ] = idxEle

        idxEle+=1

    return orderArray


dfAll = pd.DataFrame()


# Load subjects_cleaned.txt to determine which are the image and video stimuli
iS = 2
with open("subjects_cleaned.txt", "r") as f: # read line by line
    print("************"*20)
    
    for line in f:
        if iS  >0 and iS!=22: 
            print(f'current subject {iS}')

            line = line.strip()          # remove \n
            atomos = line.split(" ") 
        

            
            fileName = 'dataset/'+atomos[0][:-1]+'/eeg/'+atomos[0][:-1]+'_task-emotion_eeg.edf'


            twoFiles = False

            if os.path.exists(fileName): 
                #there is only one file
                #raw = mne.io.read_raw_edf( fileName, preload=True)

                if iS == 22:
                    raw = mne.io.read_raw_edf( fileName, preload=True)

                    annotations = raw.annotations

                    events, event_id = mne.events_from_annotations(raw) 

                else:
                    raw = pyedflib.EdfReader(fileName)
            
            else:

                #there are multiple files, and we have to place all of them together

                fileError = 1
                try:
                    fileName = 'dataset/'+atomos[0][:-1]+'/eeg/'+atomos[0][:-1]+'_task-emotion_run-01_eeg.edf'
                    # raw = mne.io.read_raw_edf( fileName, preload=True)
                    
                    if iS == 22:
                        raw = mne.io.read_raw_edf( fileName, preload=True)
                        annotations = raw.annotations

                        events, event_id = mne.events_from_annotations(raw) 

                    else:
                        raw = pyedflib.EdfReader(fileName)


                    fileError += 1
                    fileName = 'dataset/'+atomos[0][:-1]+'/eeg/'+atomos[0][:-1]+'_task-emotion_run-02_eeg.edf'
                    # raw2 = mne.io.read_raw_edf( fileName, preload=True)
                    
                    if iS == 22:
                        raw2 = mne.io.read_raw_edf( fileName, preload=True)

                        annotations2 = raw2.annotations

                        events2, event_id2 = mne.events_from_annotations(raw2)

                    else:
                        raw2 = pyedflib.EdfReader(fileName )

                    
                    twoFiles = True # there are two files

                    
                except:

                    if fileError == 1:
                        #load only file two
                        fileName = 'dataset/'+atomos[0][:-1]+'/eeg/'+atomos[0][:-1]+'_task-emotion_run-02_eeg.edf'
                        # raw = mne.io.read_raw_edf( fileName, preload=True)
                        
                        if iS == 22:
                            raw = mne.io.read_raw_edf( fileName, preload=True)

                            annotations = raw.annotations

                            events, event_id = mne.events_from_annotations(raw)
                        else:                            
                            raw = pyedflib.EdfReader(fileName )

            

            #loading events
            videoNumber = -1
            imageNumber = -1
            ratingNumber = -1
    
            for atom in atomos:
                if atom.startswith("vid-"):
                    numbers = atom.split("-")
                    videoNumber = int(numbers[1])

                if atom.startswith("ima-"):
                    numbers = atom.split("-")
                    imageNumber = int(numbers[1])

                if atom.startswith("rat"):
                    numbers = atom.split("-")
                    ratingNumber = int(numbers[1])

            if iS == 22:
                vid_code = event_id['TypeID: '+str(videoNumber)]
                ima_code = event_id['TypeID: '+str(imageNumber)]
                rat_code = event_id['TypeID: '+str(ratingNumber)]

                vid_code2 = event_id2['TypeID: '+str(videoNumber)]
                ima_code2 = event_id2['TypeID: '+str(imageNumber)]
                rat_code2 = event_id2['TypeID: '+str(ratingNumber)]

                events2Rename = events2.copy()
                events2Rename[ events2 == vid_code2 ] = vid_code
                events2Rename[ events2 == ima_code2 ] = ima_code
                events2Rename[ events2 == rat_code2 ] = rat_code

                events = np.vstack([ events, events2Rename] )

                

                eventsRename = events.copy()
                eventsRename[ events == vid_code ] = videoNumber
                eventsRename[ events == ima_code ] = imageNumber
                eventsRename[ events == rat_code ] = ratingNumber

                
            
                events = eventsRename[:, 2]
                onsets = eventsRename[:, 0]

                # print( events.shape )
                # print( events)

            else:
                onsets, durations, descriptions = raw.readAnnotations()

                onsets = onsets.tolist()
                events = [int(s.split(" ")[1]) for s in descriptions]

                if iS ==38:
                    events = [s+10 for s in events]
                    
            
            

                if twoFiles: #there are two files

                    onsets2, durations2, descriptions2 = raw2.readAnnotations() 

                    onsets2 = onsets2 + onsets[-1]
                    onsets2 = onsets2.tolist()
                    events2 = [int(s.split(" ")[1]) for s in descriptions2]

                    # print(np.asarray(events2))
                    # print(np.asarray(onsets2))

                    onsets += onsets2
                    events += events2

              
                
            
                onsets = np.asarray(onsets)
                events = np.asarray(events)

            


            timeLapsed = np.diff(onsets)
            events = events[1:] # removing first one
            onsets = onsets[1:]

          

            subEvents = events[ timeLapsed > 0 ] 
            onsets = onsets[ timeLapsed > 0 ] 
            
           
            print( videoNumber,imageNumber , ratingNumber)
            idxSubEvents = np.where( ( subEvents == videoNumber) | 
                                   ( subEvents == imageNumber) | 
                                   ( subEvents == ratingNumber) )[0] 

            subEvents = subEvents[idxSubEvents]
            subOnset = onsets[idxSubEvents]

            

            idxValid = np.zeros( len(subEvents) ).astype(bool)

            for idx in range(0, len(subEvents)-1 ):

                if ( subEvents[idx] == videoNumber or subEvents[idx] == imageNumber) and subEvents[idx+1] == ratingNumber:
                    idxValid[idx] = True
                    idxValid[idx+1] = True

            subEvents = subEvents[idxValid]
            subOnset = subOnset[idxValid]


            # now only keep videos or images
            idxSubEventsStimulli = np.where( ( subEvents == videoNumber) | 
                                   ( subEvents == imageNumber)  )[0]
            
            subEvents = subEvents[idxSubEventsStimulli]
            subOnset = subOnset[idxSubEventsStimulli]

            

            currentPos = 0
            idxValid = np.zeros( len(subEvents) ).astype(bool)
            while currentPos < len(subEvents):
                #checking next 3 are from the same stimulli
                if ( subEvents[currentPos] == subEvents[currentPos+1] and 
                    subEvents[currentPos] == subEvents[currentPos+2] ):
                    # this is good
                    idxValid[currentPos] = True
                    idxValid[currentPos+1] = True
                    idxValid[currentPos+2] = True
                    currentPos +=3
                else:
                    #this is invalid -- extra one
                    currentPos +=1

            subEvents = subEvents[idxValid]
            subOnset = subOnset[idxValid]

            print(subEvents)
            print(subOnset)
            

            if iS ==24 or iS ==53:
                subEvents = subEvents[3:]
                subOnset = subOnset[3:]


                
         
            raw.close()
            if twoFiles:
                raw2.close()

            print(f'events {len(subEvents)}')
            if len(subEvents) == 42:
                subEventsStrings = np.empty(subEvents.shape[0]).astype(str)

                subEventsStrings[ subEvents==videoNumber ]='video'
                subEventsStrings[ subEvents==imageNumber ]='image'

                subEventsStrings = np.asarray( subEventsStrings )


                nameFolder= atomos[0][:-1]
            
            
                pathFile = "dataset/"+nameFolder+"/beh/"+nameFolder+"_task-emotion_beh.tsv"

                
                if iS!= 54:
                    orderStimulus = allOrder[iS-1]

                    print(orderStimulus)

                    idxActualOrder = pairingList( reorder, orderStimulus )

                    stimulus = orderStimulus*2
                    actualOrder = np.hstack([idxActualOrder, idxActualOrder])

                else:

                    idxActualOrderImages = pairingList( reorder, sub54Ima )
                    idxActualOrderVideos = pairingList( reorder, sub54Vid )

                    stimulus = sub54Ima + sub54Vid

                    actualOrder = np.hstack([idxActualOrderImages, idxActualOrderVideos])

                df = pd.read_csv(pathFile, sep="\t")



                typeStimulus = subEventsStrings
                df['type'] = typeStimulus
                df['subject'] = [iS]*len(typeStimulus)

                

                df = df.sort_values(by=['type', 'trial_number'])
                df.reset_index(drop=True, inplace=True) # sorted by type

                # including type and order for each video type
                df['actualOrder'] = actualOrder
                df['stimulus'] = stimulus
                

                df = df.sort_values(by=['type', 'actualOrder'])

                df.rename(columns=dicMap, inplace=True)
                print(f'subject {iS}')
                print( actualOrder )
                print( df )

                dfAll = pd.concat([dfAll, df], ignore_index=True)


        iS+=1



dfAll.to_csv('allData.csv')


groupped = dfAll.groupby(['type', 'stimulus'])[['Arousal','Valence'] ].mean()
groupped = groupped.reset_index()
print(  groupped )

x = groupped.loc[ groupped['type']=='video', 'Arousal']
y = groupped.loc[ groupped['type']=='video', 'Valence'] 

z = groupped.loc[ groupped['type']=='video', 'stimulus'].values

z = np.asarray( [e[:-1] for e in z ] )

print(z)

plt.plot( x[z=='dis'], y[z=='dis'], 'o', label='dis' )
plt.plot( x[z=='fear'], y[z=='fear'], 'o', label='fear' )
plt.plot( x[z=='ins'], y[z=='ins'], 'o', label='ins' )
plt.plot( x[z=='joy'], y[z=='joy'], 'o', label='joy' )
plt.plot( x[z=='neu'], y[z=='neu'], 'o', label='neu' )
plt.plot( x[z=='sad'], y[z=='sad'], 'o', label='sad' )
plt.plot( x[z=='ten'], y[z=='ten'], 'o', label='ten' )

plt.legend() 
plt.savefig('ddd.png', dpi=300)
