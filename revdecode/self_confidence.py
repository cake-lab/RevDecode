## ###
#  IP: GHIDRA
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
#  
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
##
# Example of performing a BSim query on a single function
# @category BSim.python  

import ghidra.features.bsim.query.BSimClientFactory as BSimClientFactory
import ghidra.features.bsim.query.GenSignatures as GenSignatures
import ghidra.features.bsim.query.protocol.QueryNearest as QueryNearest
import java.util.HashSet
import hashlib
import json
import pickle
import os

DATABASE_URL = "file:/Path_to_project/Sample_binary/bsim_database/sample_binary"
OUTPUT_FILE_PATH = '/Path_to_project/Sample_binary/self_confidence_scores'
if not os.path.exists(OUTPUT_FILE_PATH):
    os.makedirs(OUTPUT_FILE_PATH)

MATCHES_PER_FUNC = 1000
SIMILARITY_BOUND = -100.0
CONFIDENCE_BOUND = -10000000.0

def query(func):
    
    url = BSimClientFactory.deriveBSimURL(DATABASE_URL)
    database = BSimClientFactory.buildClient(url,False)
    if not database.initialize():
        print database.getLastError().message
        return
    gensig = GenSignatures(False)
    gensig.setVectorFactory(database.getLSHVectorFactory())
    gensig.openProgram(currentProgram,None,None,None,None,None)
    
    gensig.scanFunction(func)

    query = QueryNearest()
    query.manage = gensig.getDescriptionManager()
    query.max = MATCHES_PER_FUNC
    query.thresh = SIMILARITY_BOUND
    query.signifthresh = CONFIDENCE_BOUND

    response = database.query(query)
    if response is None:
        print database.getLastError().message
        return
    simIter = response.result.iterator()
    matches = {}
    target_library = ""
    target_function_name = ""
    target_function = ""
    Flag = False
    while simIter.hasNext():
        sim = simIter.next()
        base = sim.getBase()
        exe = base.getExecutableRecord()
        target_library = exe.getNameExec()
        target_function_name = base.getFunctionName()
        target_function = target_library + "____" + target_function_name
        subIter = sim.iterator()
        n = 0
        while subIter.hasNext():
            Flag = True
            n = n + 1
            note = subIter.next()
            fdesc = note.getFunctionDescription()
            exerec = fdesc.getExecutableRecord()
            matched_library = exerec.getNameExec()
            matched_function_name = fdesc.getFunctionName()
            similarity_score = note.getSimilarity()
            significance_score = note.getSignificance()
            match = matched_library + "____" + matched_function_name
            if match != target_function:
                continue
            matches[match] = [similarity_score, significance_score]
            break
    if len(matches) == 0 and Flag:
        print "Not enough matches found for %s" % func.getName()
        print "Only %d matches found." % len(matches)
        # exit(1)
    gensig.dispose()
    database.close()

    return target_function, matches


funcsToQuery = java.util.HashSet()
fIter = currentProgram.getFunctionManager().getFunctionsNoStubs(True)
all_functions = {}
Library = ""
for func in fIter:
    Target_function, Matches = query(func)
    if Library == "":
        Library = Target_function.split("____")[0]
    all_functions[Target_function] = Matches


with open(OUTPUT_FILE_PATH + '/' + Library + '_all_functions_self_confidence.pkl', 'wb') as file_out:
    pickle.dump(all_functions, file_out)