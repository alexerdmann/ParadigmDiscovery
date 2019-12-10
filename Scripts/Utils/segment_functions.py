""" MICHA'S BASE AND EXPONENT IDENTIFYING FUNCTIONS """

from functools import lru_cache


def lcs(forms):
    forms = list(wf for wf in forms if wf)
    if len(forms) > 2:
        return lcs1(forms[0], lcs(forms[1:]))
    else:
        return lcs1(forms[0], forms[1])


@lru_cache(maxsize=131072)
def lcs1(f1, f2):
    if f1 == f2:
        res = f1

    elif f1 == "" or f2 == "":
        res = ""

    elif f1[-1] == f2[-1]:
        res = lcs1(f1[:-1], f2[:-1]) + f1[-1]

    else:
        l1 = lcs1(f1[:-1], f2)
        l2 = lcs1(f1, f2[:-1])
        if len(l1) > len(l2):
            res = l1
        else:
            res = l2
    return res


def getExponent(stem, forms):
    res = [None] * len(forms)
    for idx in range(len(forms)):
        form = forms[idx]
        if form != None:
            form = '<{}>'.format(form)
            # print("Getting", stem, form)
            sc, aff = getExponent1(stem, form)
            exponent = tuple([xx for xx in aff if xx not in ['<', '>', ""]])
            res[idx] = exponent
    return res


def getExponent1(stem, form, inStem=False, depth=0, verbose=False, cache={}):
    if (stem, form, inStem) in cache:
        return cache[stem, form, inStem]
    if verbose:
        print("\t" * depth, "stem", stem, "form", form, inStem)
    if len(stem) == 0:
        if verbose:
            print("\t" * depth, "->", 0, form)
        return 0, [form]
    elif len(form) == 0:
        if verbose:
            print("\t" * depth, "x")
        return 10000, [""]
    else:
        doAlign = 10000
        if stem[0] == form[0]:
            doAlign, affsAlign = getExponent1(stem[1:], form[1:], True, depth+1,
                                             verbose, cache)
            if not inStem:
                doAlign += 1
                affsAlign = [""] + affsAlign
        noAlign, affsNo = getExponent1(stem, form[1:], False, depth+1,
                                      verbose, cache)
        affsNo = [form[0] + affsNo[0]] + affsNo[1:]

        if doAlign < noAlign:
            if verbose:
                print("\t" * depth, "->", doAlign, affsAlign)

            cache[stem, form, inStem] = doAlign, affsAlign
            return doAlign, affsAlign
        if verbose:
            print("\t" * depth, "->", noAlign, affsNo)

        cache[stem, form, inStem] = noAlign, affsNo
        return noAlign, affsNo
