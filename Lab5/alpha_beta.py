# This is based on the game Nim, where each player must in turn select a pile to split.
# The result of the split must be 2 piles of different sizes

def alpha_beta_decision(state: list[int]) -> list[int]:
    infinity = float('inf')

    def max_value(state, alpha, beta) -> int:
        if is_terminal(state):
            return utility_of(state)
        v = -infinity
        for successor in successors_of(state):
            v = max(v, min_value(successor, alpha, beta))
            if v >= beta:
                return v
            alpha = min(alpha, v)
        return v

    def min_value(state, alpha, beta) -> int:
        if is_terminal(state):
            return utility_of(state)
        v = infinity

        for successor in successors_of(state):
            v = min(v, max_value(successor, alpha, beta))
            if v <= alpha:
                return v
            beta = max(beta, v)
        return v

    state = argmax(
        successors_of(state),
        lambda a: min_value(a, infinity, -infinity)
    )
    return state


def is_terminal(state: list[int]) -> bool:
    unsplittable_piles = (2, 1)
    terminal = True
    for p in state:
        if p not in unsplittable_piles:
            terminal = False
            break
    return terminal


def utility_of(state: list[int]) -> int:
    """
    Takes in a terminal state and outputs +1 if the computer has won, -1 if the player has won
    And 0 if the state is not terminal
    """
    if not is_terminal(state):
        return 0

    # Since the player starts with 1 pile, the computer wants to end on an uneven number of piles,
    # since that means that the player cannot make a move
    if len(state) % 2 == 1:
        return 1

    return -1


def successors_of(state: list[int]) -> list[list[int]]:
    successors = []

    if is_terminal(state):
        return successors

    for pile in state:
        options = split_pile_options(pile)
        for option in options:
            # Create a copy of the list to avoid altering the original
            new_state = list(state)
            new_state.remove(pile)
            for p in option:
                new_state.append(p)
            successors.append(new_state)

    return successors


def split_pile_options(pile: int) -> list[list[int]]:
    candidates = []

    # 2 would cause the loop below to output [[1,1]]
    if pile == 1 or pile == 2:
        return candidates

    # We want to include 3, but not 4 for 8 | and 4, but not 5 for 9
    # Since range is excluding og the second number, this method ensures that 4 is passed to it from 8
    # ((8+1) / 2 = 4.5; round(4.5) = 4)
    # and 5 for 9:
    # ((9+1) / 2 = 5)
    for i in range(1, round((pile + 1)/2)):
        candidates.append([i, pile - i])

    return candidates


def argmax(iterable, func):
    return max(iterable, key=func)


def computer_select_pile(state: list[int]) -> list[int]:
    new_state = alpha_beta_decision(state)
    return new_state


def user_select_pile(list_of_piles: list[int]):
    '''
    Given a list of piles, asks the user to select a pile and then a split.
    Then returns the new list of piles.
    '''
    print("\n    Current piles: {}".format(list_of_piles))

    i = -1
    while i < 0 or i >= len(list_of_piles) or list_of_piles[i] < 3:
        print("Which pile (from 1 to {}, must be > 2)?".format(len(list_of_piles)))
        i = -1 + int(input())

    print("Selected pile {}".format(list_of_piles[i]))

    max_split = list_of_piles[i] - 1

    j = 0
    while j < 1 or j > max_split or j == list_of_piles[i] - j:
        if list_of_piles[i] % 2 == 0:
            print(
                'How much is the first split (from 1 to {}, but not {})?'.format(
                    max_split,
                    list_of_piles[i] // 2
                )
            )
        else:
            print(
                'How much is the first split (from 1 to {})?'.format(max_split)
            )
        j = int(input())

    k = list_of_piles[i] - j

    new_list_of_piles = list_of_piles[:i] + [j, k] + list_of_piles[i + 1:]

    print("    New piles: {}".format(new_list_of_piles))

    return new_list_of_piles


def main():
    state = [7]

    while not is_terminal(state):
        state = user_select_pile(state)
        if not is_terminal(state):
            state = computer_select_pile(state)

    print("    Final state is {}".format(state))


if __name__ == '__main__':
    main()
