# xmatuf00 / Ján Maťufka / 222124

# selected gene that has best expression from our gene_counts.txt 
TEST = """ENSG00000197747
1;1;1;1;1;1;1;1;1;1;1;1;1
151982915;151982918;151983232;151984077;151985158;151986099;151986099;151986099;151986099;151992523;151993752;151993752;151993752
151983324;151983324;151983324;151984183;151985335;151986251;151986251;151986251;151986251;151992578;151993829;151993824;151993859
-;-;-;-;-;-;-;-;-;-;-;-;-
1012
329"""

# convert indexes into sequence that are part of introns into tuples representing intervals"
# e.g. [1,2,3,4,10,11,12,13,18,21,22,23] -> [(1,4), (10,13), (18,18), (21,23)]
def compress_to_ranges(nums):
    if not nums:
        return []
    ranges = []
    start = nums[0]
    end = nums[0]
    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1] + 1:
            end = nums[i]
        else:
            ranges.append((start, end))
            start = nums[i]
            end = nums[i]
    ranges.append((start, end))
    return ranges


words = TEST.split("\n")
# geneid = words[0]
# chrom = words[1].split(';')
starts = [int(i) for i in words[2].split(';')]
ends = [int(i) for i in words[3].split(';')]
# strands = words[4].split(';')
# length = int(words[5])
# reads = int(words[6])

intervals = [(starts[i], ends[i]) for i in range(len(starts))]
min_pos = min(starts)
max_pos = max(ends)

outside = []
for i in range(min_pos, max_pos):
    current_start = i
    current_end = i
    if all(not (start <= i <= end) for start, end in intervals):
        outside.append(i)

ranges = compress_to_ranges(outside)
print("start", min_pos)
print("end", max_pos)
print("introns", ranges)
