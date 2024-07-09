import random
from itertools import combinations
from collections import Counter, defaultdict
from treys import Card, Evaluator, Deck  # You may need to install the treys library

class PokerBot:
    def __init__(self):
        self.evaluator = Evaluator()
        self.deck = Deck()
        self.opponent_models = defaultdict(lambda: {'aggressiveness': 0.5, 'bluffing_frequency': 0.1})
        self.betting_history = []

    def convert_card(self, card):
        rank, suit = card
        suits = {'H': 'h', 'D': 'd', 'C': 'c', 'S': 's'}
        return Card.new(f'{rank}{suits[suit]}')

    def evaluate_hand(self, hand):
        treys_hand = [self.convert_card(card) for card in hand]
        rank = self.evaluator.evaluate([], treys_hand)
        return rank

    def estimate_win_probability(self, hand, community_cards, num_players, simulations=1000):
        self.deck.shuffle()
        hand_cards = [self.convert_card(card) for card in hand]
        community_cards = [self.convert_card(card) for card in community_cards]
        for card in hand_cards + community_cards:
            self.deck.cards.remove(card)

        wins = 0
        for _ in range(simulations):
            self.deck.shuffle()
            remaining_community = self.deck.draw(5 - len(community_cards))
            final_community = community_cards + remaining_community

            my_best_hand = self.evaluator.evaluate(final_community, hand_cards)

            opponent_hands = [[self.deck.draw(1), self.deck.draw(1)] for _ in range(num_players - 1)]
            opponent_best_hands = [self.evaluator.evaluate(final_community, op_hand) for op_hand in opponent_hands]

            if my_best_hand < min(opponent_best_hands):
                wins += 1

        return wins / simulations

    def update_opponent_model(self, opponent_id, action, amount):
        model = self.opponent_models[opponent_id]
        if action in ['raise', 'bet']:
            model['aggressiveness'] += 0.05
        elif action == 'fold':
            model['aggressiveness'] -= 0.05
        if action == 'bluff':
            model['bluffing_frequency'] += 0.1

        # Ensure values are within reasonable bounds
        model['aggressiveness'] = min(max(model['aggressiveness'], 0), 1)
        model['bluffing_frequency'] = min(max(model['bluffing_frequency'], 0), 1)

    def make_decision(self, hand, community_cards, num_players, position, pot, current_bet, round_name, opponent_actions):
        win_prob = self.estimate_win_probability(hand, community_cards, num_players)
        pot_odds = current_bet / (pot + current_bet)

        for opponent_id, action in opponent_actions:
            self.update_opponent_model(opponent_id, action, current_bet)

        # Calculate effective win probability considering opponents' aggressiveness and bluffing frequency
        effective_win_prob = win_prob
        for model in self.opponent_models.values():
            if model['aggressiveness'] > 0.7 and model['bluffing_frequency'] > 0.3:
                effective_win_prob += 0.05

        # Strategic decision based on win probability, pot odds, round, and opponent models
        if round_name == 'pre-flop':
            if effective_win_prob > 0.75:
                return "Raise"
            elif effective_win_prob > 0.5:
                return "Check"
            else:
                return "Fold"
        elif round_name in ['flop', 'turn', 'river']:
            if effective_win_prob > 0.7 and effective_win_prob > pot_odds:
                return "Raise"
            elif effective_win_prob > 0.4 and effective_win_prob > pot_odds:
                return "Check"
            else:
                return "Fold"

    def add_betting_action(self, player_id, action, amount):
        self.betting_history.append((player_id, action, amount))

    def print_betting_history(self):
        for record in self.betting_history:
            print(f"Player {record[0]}: {record[1]} ${record[2]}")

# Example usage
def main():
    bot = PokerBot()
    num_players = int(input("Enter number of players: "))
    position = int(input("Enter your position (0 for first, etc.): "))
    pot = float(input("Enter the current pot size: "))
    current_bet = float(input("Enter the current bet to call: "))
    round_name = input("Enter the current betting round (pre-flop, flop, turn, river): ")

    hand_input = input("Enter your hand (e.g., '14H, 13H'): ")
    hand = [(int(card[:-1]), card[-1]) for card in hand_input.split(', ')]

    community_input = input("Enter community cards if any (e.g., '2H, 3H, 4H'): ")
    if community_input:
        community_cards = [(int(card[:-1]), card[-1]) for card in community_input.split(', ')]
    else:
        community_cards = []

    # Simulating opponent actions
    opponent_actions = [
        (1, 'raise', 100),
        (2, 'call', 100),
        (3, 'fold', 0)
    ]

    for action in opponent_actions:
        bot.add_betting_action(action[0], action[1], action[2])

    decision = bot.make_decision(hand, community_cards, num_players, position, pot, current_bet, round_name, opponent_actions)
    print(f"Decision: {decision}")

    print("\nBetting History:")
    bot.print_betting_history()

if __name__ == "__main__":
    main()