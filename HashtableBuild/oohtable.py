"""
A HashTable represented as a list of lists with open hashing.
Each bucket is a list of (key,value) tuples
"""

class HashTable:
    def __init__(self, nbuckets):
        """Init with a list of nbuckets lists"""
        self.buckets = [[] for i in range(nbuckets)]
        


    def __len__(self):
        """
        number of keys in the hashable
        """
        return sum([len(x) for x in self.buckets])



    def __setitem__(self, key, value):
        """
        Perform the equivalent of table[key] = value
        Find the appropriate bucket indicated by key and then append (key,value)
        to that bucket if the (key,value) pair doesn't exist yet in that bucket.
        If the bucket for key already has a (key,value) pair with that key,
        then replace the tuple with the new (key,value).
        Make sure that you are only adding (key,value) associations to the buckets.
        The type(value) can be anything. Could be a set, list, number, string, anything!
        """
        h = hash(key) % len(self.buckets)
        for i in range(len(self.buckets[h])):
            if((self.buckets[h][i][0] == key)):
                self.buckets[h][i] = (key, value)
                break
        else:
            self.buckets[h].append((key, value))
 


    def __getitem__(self, key):
        """
        Return the equivalent of table[key].
        Find the appropriate bucket indicated by the key and look for the
        association with the key. Return the value (not the key and not
        the association!). Return None if key not found.
        """
        h = hash(key) % len(self.buckets)
        for i in range(len(self.buckets[h])):
            if(self.buckets[h][i][0] == key):
                return self.buckets[h][i][1]
        return None



    def __contains__(self, key):
        """
        check if a certain key is in the hashtable

        """
        h = hash(key) % len(self.buckets)
        for i in range(len(self.buckets[h])):
            if(self.buckets[h][i][0] == key):
                return True
        return False

        


    def __iter__(self):
        """
        iterate over all keys in the hashtable

        """
        List = self.keys()
        return iter(List)     


    def keys(self):
        """
        return all keys in the hashtable
        

        """
        ks = []
        for h in range(len(self.buckets)):
            for i in range(len(self.buckets[h])):
                ks.append(self.buckets[h][i][0])
        return ks


    def items(self):
        """
        returns all values in the hashable

        """
        vs = []
        for h in range(len(self.buckets)):
            for i in range(len(self.buckets[h])):
                vs.append(self.buckets[h][i])
        return vs


    def __repr__(self):
        """
        Return a string representing the various buckets of this table.
        The output looks like:
            0000->
            0001->
            0002->
            0003->parrt:99
            0004->
        where parrt:99 indicates an association of (parrt,99) in bucket 3.
        """
        rstr = ""
        for i in range(len(self.buckets)):
            rstr = rstr + "0"*(4-(len(str(i)))) + str(i) + "->"
            for j in range(len(self.buckets[i])):
                rstr = rstr + str(self.buckets[i][j][0]) + ":" + str(self.buckets[i][j][1])
                if (j+1 != len(self.buckets[i])):
                    rstr += ", "
            if (i+1 <= len(self.buckets)):
                rstr += "\n"
        return rstr



    def __str__(self):
        """
        Return what str(table) would return for a regular Python dict
        such as {parrt:99}. The order should be in bucket order and then
        insertion order within each bucket. The insertion order is
        guaranteed when you append to the buckets in htable_put().
        """
        return_s = "{"
        for i in range(len(self.buckets)):
            bucket = ""
            for j in range(len(self.buckets[i])):
                bucket += str(self.buckets[i][j][0])
                bucket += ":"
                bucket += str(self.buckets[i][j][1])
                bucket += (", ")
            return_s += bucket
        if(len(return_s) > 3):
            return_s = return_s[:-2]
        return_s += "}"
        return return_s




    def bucket_indexof(self, key):
        """
        You don't have to implement this, but I found it to be a handy function.

        Return the index of the element within a specific bucket; the bucket is:
        table[hashcode(key) % len(table)]. You have to linearly
        search the bucket to find the tuple containing key.
        """
        h = hash(key) % len(self.buckets)
        pass
