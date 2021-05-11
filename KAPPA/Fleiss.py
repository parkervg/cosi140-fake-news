# Deal with duplicates (Currently overwrites duplicates for each annotator)
# Start work with datapoints with three annotators
# 601 different datapoints annotated
# 186 datapoints with all 3 different annotators
# ShanChen 600; BradenLee 385; ShengLu 402

import copy

class KappaScores():
	def __init__(self):
		# self.class_dict = {"user": 0, "datapoint": 1, "contradictory_quote": 2, "exaggeration": 3, "quantitative_data": 4,
		# "evidence_lacking": 5, "dubious_reference": 6, "out_of_context": 7, "qualitative_data": 8}
		# self.line_dict = {0: "annotator", 1: "datapoint"}
		self.category_dict = {2: "Contradictory Quote", 3: "Exaggeration", 4: "Quantitative Data", 5: "Evidence Lacking",
		6: "Dubious Reference", 7: "Out of Context", 8: "Qualitative Data"}

	def category(self):
		fleiss_kappa_values = []
		cohen_shan_sheng = []
		cohen_shan_braden = []
		for n in range(2, 9):
			FK, CK_SS, CK_SB = self.fileProcessing("Data.txt", n)
			# print("\t" + self.category_dict[n] + ":", FK)
			fleiss_kappa_values.append(FK)
			cohen_shan_sheng.append(CK_SS)
			cohen_shan_braden.append(CK_SB)
		print(fleiss_kappa_values)
		average_fleiss = self.averageKappa(fleiss_kappa_values)
		print("AVERAGE FLEISS:", average_fleiss)
		print(cohen_shan_sheng)
		average_cohen_ss = self.averageKappa(cohen_shan_sheng)
		print("AVERAGE SHAN SHENG:", average_cohen_ss)
		print(cohen_shan_braden)
		average_cohen_sb = self.averageKappa(cohen_shan_braden)
		print("AVERAGE SHAN BRADEN:", average_cohen_sb)

	def fileProcessing(self, file, n):
		category_data = {}
		with open(file) as f:
			lines = f.readlines()
			for line in lines:
				line = line.split()
				# print(line)
				annotator = line[0]
				datapoint = line[1]
				if datapoint in category_data:
					# Takes care of duplicates
					if annotator in category_data[datapoint]:
						if category_data[datapoint][annotator] == 0:
							category_data[datapoint][annotator] += int(line[n])
					else:
						category_data[datapoint][annotator] = int(line[n])
				else:
					category_data[datapoint] = {annotator: int(line[n])}
		# print(category_data)
		FK = self.dictionaryPosNeg(category_data)
		CK_SS, CK_SB = self.dictionaryTwoAnnotators(category_data)
		return FK, CK_SS, CK_SB

	def dictionaryPosNeg(self, dict1):
		# shanChen = []
		# bradenLee = []
		# shengLu = []
		dictPosNeg = {}
		for datapoint in dict1:
			if len(dict1[datapoint]) == 3:
				positive = sum(dict1[datapoint].values())
				dictPosNeg[datapoint] = {"POS": positive}
				dictPosNeg[datapoint]["NEG"] = 3 - positive
			# for annotator in dict1[datapoint]:
			# 	if annotator == "SC":
			# 		shanChen.append(datapoint)
			# 	elif annotator == "BL":
			# 		bradenLee.append(datapoint)
			# 	elif annotator == "SL":
			# 		shengLu.append(datapoint)
		# print(len(shanChen))
		# print(shanChen)
		# print(len(bradenLee))
		# print(bradenLee)
		# print(len(shengLu))
		# print(shengLu)

		# Print keys to see datapoints with all three annotators
		# print(dictPosNeg.keys())
		# print(len(dictPosNeg))
		FK = self.Fleiss(dictPosNeg)
		return FK

	def Fleiss(self, dict2):
		pPos = 0
		pNeg = 0
		Ao = 0
		for datapoint in dict2:
			positive = dict2[datapoint]["POS"]
			negative = dict2[datapoint]["NEG"]
			pPos += positive
			pNeg += negative
			Pi = ((positive**2 - positive) + (negative**2 - negative)) / (3*(3-1))
			Ao += Pi
		# Pq = (counts for each category) / (numItems * annotators)
		pPos /= (len(dict2) * 3)
		pNeg /= (len(dict2) * 3)
		Ae = pPos**2 + pNeg**2
		Ao /= len(dict2)
		# (Amount of agreement above chance) / Maximum possible agreement above chance:
		K = (Ao - Ae) / (1 - Ae)
		return K

	def averageKappa(self, list1):
		total = 0
		for k in list1:
			total += k
		average = total / len(list1)
		return average

	def dictionaryTwoAnnotators(self, dict3):
		dict_shan_sheng = copy.deepcopy(dict3)
		dict_shan_braden = copy.deepcopy(dict3)
		for datapoint in dict3:
			if "BL" in dict_shan_sheng[datapoint]:
				dict_shan_sheng[datapoint].pop("BL")
			if "SL" in dict_shan_braden[datapoint]:
				dict_shan_braden[datapoint].pop("SL")
			if len(dict_shan_sheng[datapoint]) != 2:
				dict_shan_sheng.pop(datapoint)
			if len(dict_shan_braden[datapoint]) != 2:
				dict_shan_braden.pop(datapoint)
		# calculate cohens between Shan and Sheng (402)
		# print(len(dict_shan_sheng))
		CK_SS = self.Cohen(dict_shan_sheng, "SC", "SL")
		# calculate cohens between Shan and Braden (384)
		# print(len(dict_shan_braden))
		CK_SB = self.Cohen(dict_shan_braden, "SC", "BL")
		return CK_SS, CK_SB

	def Cohen(self, dict4, name1, name2):
		pos_pos = 0
		neg_neg = 0
		pos_neg = 0
		neg_pos = 0
		for datapoint in dict4:
			if dict4[datapoint][name1] == 1:
				if dict4[datapoint][name2] == 1:
					pos_pos += 1
				elif dict4[datapoint][name2] == 0:
					pos_neg += 1
			elif dict4[datapoint][name1] == 0:
				if dict4[datapoint][name2] == 1:
					neg_pos += 1
				elif dict4[datapoint][name2] == 0:
					neg_neg += 1
		Ao = (pos_pos + neg_neg) / len(dict4)
		Aek = ((((pos_pos + pos_neg) * (pos_pos + neg_pos)) / len(dict4)) + (((neg_pos + neg_neg) * (pos_neg + neg_neg)) / len(dict4))) / len(dict4)
		K = (Ao - Aek) / (1 - Aek)
		# print(pos_pos + neg_neg + pos_neg + neg_pos)
		return K


if __name__ == '__main__':
	ks = KappaScores()
	ks.category()
