def how_many_different_values_in_list(incoming_list):
    seen = []
    dupes = []
    for x in incoming_list:
        if x in seen:
            dupes.append(x)
        else:
            seen.append(x)

    return len(seen)
