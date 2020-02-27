#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Metaphone algorithm from Jellyfish library
# String Similarity and distance algorithm from strsimpy library


# In[2]:


class StringDistance:
    def distance(self, s0, s1):
        raise NotImplementedError()

class NormalizedStringDistance(StringDistance):
    def distance(self, s0, s1):
        raise NotImplementedError()

class MetricStringDistance(StringDistance):
    def distance(self, s0, s1):
        raise NotImplementedError()
        
class StringSimilarity:
    def similarity(self, s0, s1):
        raise NotImplementedError()

class NormalizedStringSimilarity(StringSimilarity):
    def similarity(self, s0, s1):
        raise NotImplementedError()
        


# ## Levenshtein

# In[3]:


class Levenshtein(MetricStringDistance):

    def distance(self, s0, s1):
        if s0 is None:
            raise TypeError("Argument s0 is NoneType.")
        if s1 is None:
            raise TypeError("Argument s1 is NoneType.")
        if s0 == s1:
            return 0.0
        if len(s0) == 0:
            return len(s1)
        if len(s1) == 0:
            return len(s0)

        v0 = [0] * (len(s1) + 1)
        v1 = [0] * (len(s1) + 1)

        for i in range(len(v0)):
            v0[i] = i

        for i in range(len(s0)):
            v1[0] = i + 1
            for j in range(len(s1)):
                cost = 1
                if s0[i] == s1[j]:
                    cost = 0
                v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
            v0, v1 = v1, v0

        return v0[len(s1)]


# In[4]:


class NormalizedLevenshtein(NormalizedStringDistance, NormalizedStringSimilarity):

    def __init__(self):
        self.levenshtein = Levenshtein()

    def distance(self, s0, s1):
        if s0 is None:
            raise TypeError("Argument s0 is NoneType.")
        if s1 is None:
            raise TypeError("Argument s1 is NoneType.")
        if s0 == s1:
            return 0.0
        m_len = max(len(s0), len(s1))
        if m_len == 0:
            return 0.0
        return self.levenshtein.distance(s0, s1) / m_len

    def similarity(self, s0, s1):
        return 1.0 - self.distance(s0, s1)


# ## NGram

# In[5]:


class NGram(NormalizedStringDistance):

    def __init__(self, n=2):
        self.n = n

    def distance(self, s0, s1):
        if s0 is None:
            raise TypeError("Argument s0 is NoneType.")
        if s1 is None:
            raise TypeError("Argument s1 is NoneType.")
        if s0 == s1:
            return 0.0

        special = '\n'
        sl = len(s0)
        tl = len(s1)

        if sl == 0 or tl == 0:
            return 1.0

        cost = 0
        if sl < self.n or tl < self.n:
            for i in range(min(sl, tl)):
                if s0[i] == s1[i]:
                    cost += 1
            return 1.0 * cost / max(sl, tl)

        sa = [''] * (sl + self.n - 1)

        for i in range(len(sa)):
            if i < self.n - 1:
                sa[i] = special
            else:
                sa[i] = s0[i - self.n + 1]

        p = [0.0] * (sl + 1)
        d = [0.0] * (sl + 1)
        t_j = [''] * self.n
        for i in range(sl + 1):
            p[i] = 1.0 * i

        for j in range(1, tl + 1):
            if j < self.n:
                for ti in range(self.n - j):
                    t_j[ti] = special
                for ti in range(self.n - j, self.n):
                    t_j[ti] = s1[ti - (self.n - j)]
            else:
                t_j = s1[j - self.n:j]

            d[0] = 1.0 * j
            for i in range(sl + 1):
                cost = 0
                tn = self.n
                for ni in range(self.n):
                    if sa[i - 1 + ni] != t_j[ni]:
                        cost += 1
                    elif sa[i - 1 + ni] == special:
                        tn -= 1
                ec = cost / tn
                d[i] = min(d[i - 1] + 1, p[i] + 1, p[i - 1] + ec)
            p, d = d, p

        return p[sl] / max(tl, sl)


# ## JaroWinkler

# In[6]:


class JaroWinkler(NormalizedStringSimilarity, NormalizedStringDistance):

    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.three = 3
        self.jw_coef = 0.1

    def get_threshold(self):
        return self.threshold

    def similarity(self, s0, s1):
        if s0 is None:
            raise TypeError("Argument s0 is NoneType.")
        if s1 is None:
            raise TypeError("Argument s1 is NoneType.")
        if s0 == s1:
            return 1.0
        mtp = self.matches(s0, s1)
        m = mtp[0]
        if m == 0:
            return 0.0
        j = (m / len(s0) + m / len(s1) + (m - mtp[1]) / m) / self.three
        jw = j
        if j > self.get_threshold():
            jw = j + min(self.jw_coef, 1.0 / mtp[self.three]) * mtp[2] * (1 - j)
        return jw

    def distance(self, s0, s1):
        return 1.0 - self.similarity(s0, s1)

    @staticmethod
    def matches(s0, s1):
        if len(s0) > len(s1):
            max_str = s0
            min_str = s1
        else:
            max_str = s1
            min_str = s0
        ran = int(max(len(max_str) / 2 - 1, 0))
        match_indexes = [-1] * len(min_str)
        match_flags = [False] * len(max_str)
        matches = 0
        for mi in range(len(min_str)):
            c1 = min_str[mi]
            for xi in range(max(mi - ran, 0), min(mi + ran + 1, len(max_str))):
                if not match_flags[xi] and c1 == max_str[xi]:
                    match_indexes[mi] = xi
                    match_flags[xi] = True
                    matches += 1
                    break

        ms0, ms1 = [0] * matches, [0] * matches
        si = 0
        for i in range(len(min_str)):
            if match_indexes[i] != -1:
                ms0[si] = min_str[i]
                si += 1
        si = 0
        for j in range(len(max_str)):
            if match_flags[j]:
                ms1[si] = max_str[j]
                si += 1
        transpositions = 0
        for mi in range(len(ms0)):
            if ms0[mi] != ms1[mi]:
                transpositions += 1
        prefix = 0
        for mi in range(len(min_str)):
            if s0[mi] == s1[mi]:
                prefix += 1
            else:
                break
        return [matches, int(transpositions / 2), prefix, len(max_str)]


# ## Metaphone

# In[7]:


import unicodedata
def _check_type(s):
    if not isinstance(s, str):
        raise TypeError('expected str or unicode, got %s' % type(s).__name__)
        
def _normalize(s):
    return unicodedata.normalize('NFKD', s)

def metaphone(s):
    _check_type(s)

    result = []

    s = _normalize(s.lower())

    # skip first character if s starts with these
    if s.startswith(('kn', 'gn', 'pn', 'ac', 'wr', 'ae')):
        s = s[1:]

    i = 0

    while i < len(s):
        c = s[i]
        next = s[i+1] if i < len(s)-1 else '*****'
        nextnext = s[i+2] if i < len(s)-2 else '*****'

        # skip doubles except for cc
        if c == next and c != 'c':
            i += 1
            continue

        if c in 'aeiou':
            if i == 0 or s[i-1] == ' ':
                result.append(c)
        elif c == 'b':
            if (not (i != 0 and s[i-1] == 'm')) or next:
                result.append('b')
        elif c == 'c':
            if next == 'i' and nextnext == 'a' or next == 'h':
                result.append('x')
                i += 1
            elif next in 'iey':
                result.append('s')
                i += 1
            else:
                result.append('k')
        elif c == 'd':
            if next == 'g' and nextnext in 'iey':
                result.append('j')
                i += 2
            else:
                result.append('t')
        elif c in 'fjlmnr':
            result.append(c)
        elif c == 'g':
            if next in 'iey':
                result.append('j')
            elif next not in 'hn':
                result.append('k')
            elif next == 'h' and nextnext and nextnext not in 'aeiou':
                i += 1
        elif c == 'h':
            if i == 0 or next in 'aeiou' or s[i-1] not in 'aeiou':
                result.append('h')
        elif c == 'k':
            if i == 0 or s[i-1] != 'c':
                result.append('k')
        elif c == 'p':
            if next == 'h':
                result.append('f')
                i += 1
            else:
                result.append('p')
        elif c == 'q':
            result.append('k')
        elif c == 's':
            if next == 'h':
                result.append('x')
                i += 1
            elif next == 'i' and nextnext in 'oa':
                result.append('x')
                i += 2
            else:
                result.append('s')
        elif c == 't':
            if next == 'i' and nextnext in 'oa':
                result.append('x')
            elif next == 'h':
                result.append('0')
                i += 1
            elif next != 'c' or nextnext != 'h':
                result.append('t')
        elif c == 'v':
            result.append('f')
        elif c == 'w':
            if i == 0 and next == 'h':
                i += 1
                if nextnext in 'aeiou' or nextnext == '*****':
                    result.append('w')
            elif next in 'aeiou':
                result.append('w')
        elif c == 'x':
            if i == 0:
                if next == 'h' or (next == 'i' and nextnext in 'oa'):
                    result.append('x')
                else:
                    result.append('s')
            else:
                result.append('k')
                result.append('s')
        elif c == 'y':
            if next in 'aeiou':
                result.append('y')
        elif c == 'z':
            result.append('s')
        elif c == ' ':
            if len(result) > 0 and result[-1] != ' ':
                result.append(' ')

        i += 1

    return ''.join(result).upper()


# In[9]:


s1 = input("input the first word \t :")
s2 = input("input teh second word \t :")
print ('\nThe result of metaphone for \'%s\' \t : %.4s' %(s1, metaphone(s1)))
print ('The result of metaphone for \'%s\' \t : %.4s' %(s2, metaphone(s2)))

print ('\nDistance between \'%s\' and \'%s\' using Levensthein Distance \t \t \t : %.4f' % (s1, s2, Levenshtein().distance(s1, s2)))
print ('Distance after metaphone between \'%s\' and \'%s\' using Levensthein Distance \t : %.4f' % (s1, s2, Levenshtein().distance(metaphone(s1), metaphone(s2))))

print ('\nDistance between \'%s\' and \'%s\' using NormalizedLevensthein Distance \t \t \t : %.4f' % (s1, s2, NormalizedLevenshtein().distance(s1, s2)))
print ('distance after metaphone between \'%s\' and \'%s\' using NormalizedLevensthein Distance \t : %.4f' % (s1, s2, NormalizedLevenshtein().distance(metaphone(s1), metaphone(s2))))

print ('\nDistance between \'%s\' and \'%s\' using NGram Distance \t \t \t : %.4f' % (s1, s2, NGram().distance(s1, s2))) 
print ('distance after metaphone between \'%s\' and \'%s\' using NGram Distance \t : %.4f' % (s1, s2, NGram().distance(metaphone(s1), metaphone(s2))))

print ('\nDistance between \'%s\' and \'%s\' using JaroWinkler Distance \t \t \t \t : %.4f' % (s1, s2, JaroWinkler().distance(s1, s2))) 
print ('distance after metaphone between \'%s\' and \'%s\' using JaroWinkler Distance \t \t : %.4f' % (s1, s2, JaroWinkler().distance(metaphone(s1), metaphone(s2))))

print ('\nSimilarity between \'%s\' and \'%s\' using NormalizedLevensthein Similarity \t \t \t \t : %.4f' % (s1, s2, NormalizedLevenshtein().similarity(s1, s2)))
print ('Similarity after metaphone between \'%s\' and \'%s\' using NormalizedLevensthein Similarity  \t : %.4f' % (s1, s2, NormalizedLevenshtein().similarity(metaphone(s1), metaphone(s2))))

print ('\nSimilarity between \'%s\' and \'%s\' using JaroWinkler Similarity \t \t \t : %.4f' % (s1, s2, JaroWinkler().similarity(s1, s2)))
print ('Similarity after metaphone between \'%s\' and \'%s\' using JaroWinkler Similarity \t : %.4f' % (s1, s2, JaroWinkler().similarity(metaphone(s1), metaphone(s2))))


# In[ ]:




