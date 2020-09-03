from pysc2.lib.units import Protoss, Terran, Zerg
from pysc2.lib.upgrades import Upgrades
import numpy as np

class GlobalParser:

    def __init__(self, player_race, enemy_race):
        self.id_alias = {
            Protoss.DisruptorPhased : Protoss.Disruptor,
            Protoss.ObserverSurveillanceMode : Protoss.Observer,
            Protoss.WarpPrismPhasing : Protoss.WarpPrism,
            Terran.GhostAlternate : Terran.Ghost,
            Terran.GhostNova : Terran.Ghost,
            Terran.LiberatorAG : Terran.Liberator,
            Terran.SiegeTankSieged : Terran.SiegeTank,
            Terran.SupplyDepotLowered : Terran.SupplyDepot,
            Terran.ThorHighImpactMode : Terran.Thor,
            Terran.VikingAssault : Terran.VikingFighter,
            Zerg.OverseerOversightMode : Zerg.Overseer
        }
        self.player_unit_map = self.map_unit_ids(player_race)
        self.enemy_unit_map = self.map_unit_ids(enemy_race)
        self.upgrade_map = self.map_upgrade_ids()

        # self.player_upgrades = [0,0,0]
        # self.enemy_upgrades = [0,0,0]
        self.enemy_tags = [set() for _ in self.enemy_unit_map]

    def map_unit_ids(self, race):
        id_map = {}
        id_idx = 0

        race_map = {'Terran': Terran, 'Zerg': Zerg, 'Protoss': Protoss}
        race_enum = race_map[race]

        for unit, unit_id in race_enum.__members__.items():

            if unit_id.value in self.id_alias:
                continue
            elif unit.endswith('Burrowed'):
                unit = unit.replace('Burrowed','')
            elif unit.endswith('Uprooted'):
                unit = unit.replace('Uprooted','')
            elif unit.endswith('Flying'):
                unit = unit.replace('Flying','')
            else:
                id_map[unit_id] = id_idx
                id_idx += 1
                continue

            self.id_alias[unit_id] = race_enum[unit]

        return id_map
    
    def map_upgrade_ids(self):

        id_map = {}
        id_idx = 0

        for _, upgrade_id in Upgrades.__members__.items():
            id_map[upgrade_id] = id_idx
            id_idx += 1

        return id_map

    def extract(self, obs):

        state = np.array([
            obs.game_loop,
            obs.score.score,
            obs.score.score_details.idle_production_time,
            obs.score.score_details.idle_worker_time,
            obs.score.score_details.total_value_units,
            obs.score.score_details.total_value_structures,
            obs.score.score_details.killed_value_units,
            obs.score.score_details.killed_value_structures,
            obs.score.score_details.collected_minerals,
            obs.score.score_details.collected_vespene,
            obs.score.score_details.collection_rate_minerals,
            obs.score.score_details.collection_rate_vespene,
            obs.score.score_details.spent_minerals,
            obs.score.score_details.spent_vespene,
            obs.player_common.minerals,
            obs.player_common.vespene,
            obs.player_common.food_cap,
            obs.player_common.food_used,
            obs.player_common.food_army,
            obs.player_common.food_workers,
            obs.player_common.idle_worker_count,
            obs.player_common.army_count,
            obs.player_common.warp_gate_count
        ])

        upgrades = self.get_upgrades(obs)
        allied_alive, allied_hp = self.get_allied_alive(obs)
        allied_construction, allied_percent = self.get_allied_construction(obs)
        enemy_visible, enemy_hp = self.get_enemy_visible(obs)
        enemy_construction, enemy_percent = self.get_enemy_construction(obs)
        enemy_killed = self.get_enemy_killed(obs)
        # enemy_seen = self.get_enemy_seen(obs)

        return np.concatenate((
            state, upgrades,
            allied_alive, allied_hp, 
            allied_construction, allied_percent, 
            enemy_visible, enemy_hp, 
            enemy_construction, enemy_percent,
            enemy_killed))

    def get_feature_list(self):
        feature_list = [
            'frame',
            'score',
            'idle_production_time',
            'idle_worker_time',
            'total_value_units',
            'total_value_structures',
            'killed_value_units',
            'killed_value_structures',
            'collected_minerals',
            'collected_vespene',
            'collection_rate_minerals',
            'collection_rate_vespene',
            'spent_minerals',
            'spent_vespene',
            'minerals',
            'vespene',
            'food_cap',
            'food_used',
            'food_army',
            'food_workers',
            'idle_worker_count',
            'army_count',
            'warp_gate_count'
        ]

        for research in sorted(self.upgrade_map, key = self.upgrade_map.get):
            feature_list.append(research.name)

        for player, unit_map in [('player', self.player_unit_map), ('enemy', self.enemy_unit_map)]:
            for feature in ['unit', 'hp', 'construction', 'percent']:
                for unit in sorted(unit_map, key = unit_map.get):
                    feature_list.append(f"{player}_{feature}_{unit.name}")

        for unit in sorted(self.enemy_unit_map, key = self.enemy_unit_map.get):
            feature_list.append(f"enemy_killed_{unit.name}")

        return feature_list

    def get_upgrades(self, obs):
        upgrades = np.zeros(len(self.upgrade_map))

        for upgrade_id in obs.raw_data.player.upgrade_ids:
            upgrades[self.upgrade_map[upgrade_id]] += 1

        return upgrades

    def get_unit_idx(self, type_map, unit_type):
        if unit_type in self.id_alias:
            unit_type = self.id_alias[unit_type]
        
        if unit_type in type_map:
            return type_map[unit_type]
        else:
            print('Unknown unit (neural parasite?):', unit_type)
            return None

    def get_allied_alive(self, obs):
        count = np.zeros(len(self.player_unit_map))
        hp = np.zeros(len(self.player_unit_map))

        for unit in obs.raw_data.units:
            if unit.alliance == 1 and unit.build_progress >= 1:
                idx = self.get_unit_idx(self.player_unit_map, unit.unit_type)
                if idx is not None:
                    count[idx] += 1
                    hp[idx] += (unit.health + unit.shield) / (unit.health_max + unit.shield_max)
                    # self.player_upgrades[0] = max(self.player_upgrades[0], unit.attack_upgrade_level)
                    # self.player_upgrades[1] = max(self.player_upgrades[1], unit.armor_upgrade_level)
                    # self.player_upgrades[2] = max(self.player_upgrades[2], unit.shield_upgrade_level)

                    for passenger in unit.passengers:
                        idx = self.get_unit_idx(self.player_unit_map, passenger.unit_type)
                        if idx is not None:
                            count[idx] += 1
                            hp[idx] += (passenger.health + passenger.shield) / (passenger.health_max + passenger.shield_max)

        for idx, num in np.ndenumerate(count):
            if num > 0:
                hp[idx] /= num

        return (count, hp)

    def get_allied_construction(self, obs):
        count = np.zeros(len(self.player_unit_map))
        percentage = np.zeros(len(self.player_unit_map))

        for unit in obs.raw_data.units:
            if unit.alliance == 1 and unit.build_progress < 1:
                idx = self.get_unit_idx(self.player_unit_map, unit.unit_type)
                if idx is not None:
                    count[idx] += 1
                    percentage[idx] += unit.build_progress

        for idx, num in np.ndenumerate(count):
            if num > 0:
                percentage[idx] /= num

        return (count, percentage)

    def get_enemy_visible(self, obs):
        count = np.zeros(len(self.enemy_unit_map))
        hp = np.zeros(len(self.enemy_unit_map))

        for unit in obs.raw_data.units:
            if unit.alliance == 4 and unit.display_type == 1 and unit.build_progress >= 1:
                idx = self.get_unit_idx(self.enemy_unit_map, unit.unit_type)
                if idx is not None:
                    count[idx] += 1
                    hp[idx] += (unit.health + unit.shield) / (unit.health_max + unit.shield_max)
                    self.enemy_tags[idx].add(unit.tag)

                    # self.enemy_upgrades[0] = max(self.enemy_upgrades[0], unit.attack_upgrade_level)
                    # self.enemy_upgrades[1] = max(self.enemy_upgrades[1], unit.armor_upgrade_level)
                    # self.enemy_upgrades[2] = max(self.enemy_upgrades[2], unit.shield_upgrade_level)

        for idx, num in np.ndenumerate(count):
            if num > 0:
                hp[idx] /= num

        return (count, hp)

    def get_enemy_construction(self, obs):
        count = np.zeros(len(self.enemy_unit_map))
        percentage = np.zeros(len(self.enemy_unit_map))

        for unit in obs.raw_data.units:
            if unit.alliance == 4 and unit.display_type == 1 and unit.build_progress < 1:
                idx = self.get_unit_idx(self.enemy_unit_map, unit.unit_type)
                if idx is not None:
                    count[idx] += 1
                    percentage[idx] += unit.build_progress
                    self.enemy_tags[idx].add(unit.tag)

        for idx, num in np.ndenumerate(count):
            if num > 0:
                percentage[idx] /= num

        return (count, percentage)

    def get_enemy_killed(self, obs):

        dead_tags = set(obs.raw_data.event.dead_units)

        count = np.zeros(len(self.enemy_unit_map))

        for idx, tags in enumerate(self.enemy_tags):
            remaining_tags = tags - dead_tags
            count[idx] = len(tags) - len(remaining_tags)

            # if count[idx] > 0:

            #     for a, b in self.enemy_unit_map.items():
            #         if b == idx:
            #             name = a.name

            #     print(f"{name} killed-{count[idx]} remaining-{len(remaining_tags)}")

            self.enemy_tags[idx] = remaining_tags

        return count

    # Doesn't work currently, need to figure out how to turn off fog of war to work this
    # def get_enemy_seen(self, obs): 
    #     count = np.zeros(len(self.enemy_unit_map))

    #     current_tags = [set() for _ in self.enemy_unit_map]

    #     for unit in obs.raw_data.units:
    #         if unit.alliance == 4:

    #             idx = self.enemy_unit_map[self.id_alias[unit.unit_type] if unit.unit_type in self.id_alias else unit.unit_type]

    #             if unit.display_type == 1:
    #                 self.enemy_tags[idx].add(unit.tag)

    #             print('Hidden unit', unit.unit_type, unit.tag, unit.display_type)

    #             current_tags[idx].add(unit.tag)

    #     for idx in range(len(self.enemy_unit_map)):
    #         # Union of seen units and all current enemy units gets a count of all the units that have been seen and have not died yet
    #         self.enemy_tags[idx] = self.enemy_tags[idx].intersection(current_tags[idx]) 
    #         count[idx] = len(self.enemy_tags[idx])

    #     return count