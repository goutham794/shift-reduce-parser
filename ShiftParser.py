from typing import List
from collections import namedtuple
import pandas as pd

parse_step = namedtuple('parse_step', ['Input_Stack', 'Rule_used', 'State_Stack', 'Backtrack'])
backtrack_step = namedtuple('backtrack_step', ['Rule_num', 'Input_Stack', 'State_Stack'])

class Symbol():
    def __init__(self, symbol: str, type: str):
        self.symbol = symbol
        assert (type == "NonTerminal") | (type=="Terminal") 
        self.type = type
    
    def __repr__(self):
        return self.symbol
    
    def __str__(self):
        return self.symbol


class Grammar_Rule():
    def __init__(self, lhs: Symbol, rhs: List[Symbol]):
        self.lhs, self.rhs = lhs, rhs
    
    def __repr__(self):
        return f"{self.lhs} -> {self.rhs}"

class Grammar():
    def __init__(self, grammar_rules: List[Grammar_Rule]):
        self.rules = grammar_rules        
    
    def __len__(self):
        return len(self.rules)

class PNA_State_Rule(Grammar_Rule):
    def __init__(self, rule: Grammar_Rule, dot_index):
        super().__init__(rule.lhs, rule.rhs)
        self.dot_index = dot_index
    
    def __repr__(self):
        return f"{self.lhs} -> {self.rhs[:self.dot_index] + ['.'] + self.rhs[self.dot_index:]}"
    
    def __eq__(self, other):
        return (self.lhs == other.lhs) & (self.rhs == other.rhs) & (self.dot_index == other.dot_index)

class PNA:
    def __init__(self, grammar: Grammar):
        self.pna_states = []
        self.grammar = grammar
        self.input_stack = []
        self.state_stack = [0]
        self.backtrack_stack = []
        self.parse_steps = []
    
    def add_state(self, state):
        self.pna_states.append(state)
    
    def _get_state_with_same_starting_rules(self, rules: List[PNA_State_Rule]):
        for state in self.pna_states:
            if state.starting_rules == rules:
                return state
        return None
    
    @property
    def current_state(self):
        return self.pna_states[self.state_stack[0]]

    @property
    def num_incomplete_states(self):
        count = 0
        for state in self.pna_states:
            if not state.is_complete:
                count += 1
        return count
    
    def construct(self):
        while self.num_incomplete_states != 0:
            self.create_transitions_reductions_for_all_states()
        
    
    def create_transitions_reductions_for_all_states(self):
        for state in self.pna_states:
            if not state.is_complete:
                state._create_transition_rules()
                state._create_reduction_rules()
                
    def display_all_states(self):
        for i, state in enumerate(self.pna_states):
            print(f"State {i}")
            print(".................")
            print("Rules")
            for i in range(len(state.all_state_rules)): print(state.all_state_rules[i])
            print("\n")
            print("Transitions")
            for i in range(len(state.transitions)): print(state.transitions[i])
            print("\n")
            print("Reductions")
            print(state.reduction_rules)
            print("\n")
            print("\n")
    
    def backtrack_step(self):
        self.input_stack = self.backtrack_stack[0].Input_Stack
        self.state_stack = self.backtrack_stack[0].State_Stack
        reduction_rule = self.current_state.reduction_rules[0]
        self.input_stack.insert(0, str(self.current_state.reduction_rules[1].lhs))
        for _ in range(len(self.current_state.reduction_rules[1].rhs)):
            _ = self.state_stack.pop(0)
        
        backtrack = backtrack_step("",[],[])
        _ = self.backtrack_stack.pop(0)
        return parse_step(self.input_stack.copy(), reduction_rule, self.state_stack.copy(), backtrack)
    
    def parse_sentence(self, sentence: List[str]):
        self.input_stack = sentence
        step = parse_step(self.input_stack.copy(), "", self.state_stack.copy(),"")
        while(self.input_stack!=['S']):
            try:
                next_input_symbol = self.input_stack[0]
            except IndexError:
                next_input_symbol = ''
            try:
                step = self.execute_step(next_input_symbol)
            except IndexError:
                step = self.backtrack_step()
            self.parse_steps.append(step)
        return pd.DataFrame(self.parse_steps)
    
    def execute_step(self, input_symbol):
        transition_state = self.current_state.get_destination_state_from_transition_symbol(input_symbol)
        backtrack = backtrack_step("",[],[])
        if transition_state == None:
            reduction_rule = self.current_state.reduction_rules[0]
            self.input_stack.insert(0, str(self.current_state.reduction_rules[1].lhs))
            for _ in range(len(self.current_state.reduction_rules[1].rhs)):
                _ = self.state_stack.pop(0)
        else:
            if self.current_state.reduction_rules != tuple():
                backtrack = backtrack_step(self.current_state.reduction_rules[0], self.input_stack.copy(), self.state_stack.copy())
                self.backtrack_stack.insert(0, backtrack)

            _ = self.input_stack.pop(0)
            reduction_rule = ""
            self.state_stack.insert(0, self.pna_states.index(transition_state))
        return parse_step(self.input_stack.copy(), reduction_rule, self.state_stack.copy(), backtrack)


        
class PNA_State():
    def __init__(self, starting_rules: List[PNA_State_Rule], parent_PNA: PNA):
        self.starting_rules, self.parent_PNA = starting_rules, parent_PNA
        self.parent_PNA.pna_states.append(self)        
        self.other_rules = []
        self._expand_rules()
        while (True):
            current_num_rules = len(self.other_rules)            
            self._expand_rules()
            if current_num_rules == len(self.other_rules):
                break
        self.transitions = []
        self.reduction_rules = tuple()
    
    def __repr__(self):
        return f"State_{self.parent_PNA.pna_states.index(self)}"
    
    @property
    def available_transitions(self):
        return [transition_rule.transition_symbol for transition_rule in self.transitions]
    
    @property
    def all_state_rules(self):
        return self.starting_rules + self.other_rules
    
    @property
    def is_complete(self):
        return (self.transitions!=[]) | (self.reduction_rules!=tuple())
    
    def get_destination_state_from_transition_symbol(self, symbol):
        for transition_rule in self.transitions:
            if str(transition_rule.transition_symbol) == symbol:
                return transition_rule.to_state
        return None

    
    def _expand_rules(self):
        for rule in self.all_state_rules:
            if rule.dot_index < len(rule.rhs):
                symbol_after_dot = rule.rhs[rule.dot_index]
                if symbol_after_dot.type == "NonTerminal":
                    for grammar_rule in self.parent_PNA.grammar.rules:
                        if symbol_after_dot == grammar_rule.lhs:
                            new_rule = PNA_State_Rule(grammar_rule, 0)
                            if new_rule not in self.other_rules:
                                self.other_rules.append(new_rule)
    
    def _create_transition_rules(self):
        rules_to_create = {}
        for rule in self.starting_rules + self.other_rules:
            if rule.dot_index < len(rule.rhs):
                if rule.rhs[rule.dot_index] not in self.available_transitions:                    
                    if rule.rhs[rule.dot_index] not in rules_to_create.keys():
                        rules_to_create[rule.rhs[rule.dot_index]] = [PNA_State_Rule(rule, rule.dot_index+1)]
                    else:
                        rules_to_create[rule.rhs[rule.dot_index]].append(PNA_State_Rule(rule, rule.dot_index+1))
        for symbol, rules in rules_to_create.items():            
            if self.parent_PNA._get_state_with_same_starting_rules(rules) == None:
                new_state = PNA_State(rules, self.parent_PNA)
                self.transitions.append(Transition_Rule(symbol, new_state))
            else:
                self.transitions.append(Transition_Rule(symbol, self.parent_PNA._get_state_with_same_starting_rules(rules)))
    
    def _create_reduction_rules(self):
        for rule in self.starting_rules + self.other_rules:
            if rule.dot_index == len(rule.rhs):
                for i,grammar_rule in enumerate(self.parent_PNA.grammar.rules):
                    if (rule.lhs == grammar_rule.lhs) & (rule.rhs == grammar_rule.rhs):
                        self.reduction_rules = (f"Rule_{i+1}",grammar_rule)
                        return
    

class Transition_Rule():
    def __init__(self, transition_symbol: Symbol, to_state: PNA_State):
        self.transition_symbol, self.to_state = transition_symbol, to_state
    
    def __repr__(self):
        return f"{self.transition_symbol} => {self.to_state}"



            