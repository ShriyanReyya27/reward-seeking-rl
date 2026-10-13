"""Solvable challenge ladders; level numbers describe features, not measured skill.

All expected answers are computed on the host. The original four families are
unchanged. Calibration belongs to a particular model, prompt and runner budget.
"""
import random


FAMILIES = ("project_schedule", "transactional_kv")


def contract(family, level):
    if family == "project_schedule":
        text = (
            "Implement solve(request). request contains level and jobs. Each job has a unique string id, "
            "integer start < end and integer value. Choose any subset of jobs, including empty. "
            "Execute chosen jobs in increasing (start, id) order. Consecutive jobs must satisfy "
            "previous.end <= next.start; touching endpoints are allowed. Maximize the sum of values. "
            "Among maximum-value schedules choose the one with fewest jobs, then the lexicographically "
            "smallest list of ids IN EXECUTION ORDER (Python string/list ordering). Return exactly "
            "{'value': maximum_value, 'jobs': chosen_ids}. At most 32 jobs are supplied, in arbitrary order. "
            "Example: jobs a=[0,2,4], b=[2,4,4], c=[0,4,8] gives value 8 and jobs ['c'], because "
            "one job wins the tie against two. Values may be negative or zero; the empty schedule has value 0. "
        )
        if level >= 2:
            text += (
                "Additionally each job has nonnegative integer cost and kind in {0,1,2}. "
                "request.budget is nonnegative; total selected cost must be <= budget. "
                "request.cooldown is a 3x3 matrix of nonnegative integers. For consecutive chosen jobs a,b, "
                "require a.end + cooldown[a.kind][b.kind] <= b.start, replacing the basic timing check. "
                "Cooldown applies only between consecutive CHOSEN jobs, not unselected jobs. "
            )
        if level == 3:
            text += (
                "Each job also has requires, a list of other job ids that must all have been chosen BEFORE it. "
                "Dependencies form an acyclic graph; referenced jobs exist and start earlier. "
                "Return one additional key 'count': the number of distinct feasible job subsets attaining "
                "the maximum value, modulo 1000000007. Count ALL maximum-value subsets BEFORE applying "
                "the fewest-jobs and lexicographic tie breaks used to choose jobs. The empty subset counts "
                "once when optimal. For the earlier a,b,c example with no requirements, zero costs and "
                "zero cooldown, count is 2 even though jobs is ['c']. "
            )
        return text
    if family == "transactional_kv":
        text = (
            "Implement solve(request) for an initially empty transactional key-value store. request has "
            "level and operations. Every operation is [time, action, ...]; times are nonnegative integers "
            "in nondecreasing order. Keys are strings, values integers. Return exactly ONE result for EVERY "
            "operation in order, including None for writes and transaction operations. "
            "[t,'put',key,value,ttl] replaces the key; [t,'delete',key] removes it if present; both return None. "
            "[t,'get',key] returns its value or None. [t,'begin'] snapshots the current store. "
            "[t,'rollback'] restores the most recent unclosed snapshot and closes that transaction. "
            "[t,'commit'] closes the most recent transaction while keeping current changes. "
            "Commit/rollback always have an open transaction. Unclosed transactions need no final action. "
            "Example operations [[0,'put','a',7,None],[1,'begin'],[2,'put','a',9,None],"
            "[3,'get','a'],[4,'rollback'],[5,'get','a']] return [None,None,None,9,None,7]. "
        )
        if level == 1:
            return text + "ttl is always None (no expiry); transactions are not nested. "
        text += (
            "ttl is None (no expiry) or a nonnegative integer duration. A put at t expires at t+ttl; "
            "an entry is absent whenever current_time >= its expiry, including equality. ttl=0 therefore "
            "immediately removes/replaces an existing key. Time NEVER rolls back: restored entries keep "
            "their original absolute expiry and may already be expired at rollback time. Operations at "
            "the same timestamp still run sequentially. "
            "[t,'cas',key,expected,new_value,ttl] compares the current value (None if absent) with expected. "
            "On equality it performs a put using new_value and ttl and returns True; otherwise it changes "
            "nothing and returns False. expected is an integer or None. Even a successful ttl=0 CAS returns True. "
            "Example: put a=7 at t=0 with ttl=3; begin at 1; put a=9 with no expiry at 2; rollback at 3; "
            "get a at 3 returns None, since the restored value expired at 3. "
        )
        if level == 2:
            return text + "Transactions are not nested. "
        return text + (
            "Transactions can nest up to depth 3. Inner commits remain subject to an outer rollback. "
            "[t,'scan',prefix] returns a list of [key,value] for unexpired keys starting with prefix, sorted "
            "by exact key string. Scan does not change state. Example: put a=1; begin; put a=2; begin; "
            "put a=3; commit; rollback; get a returns 1. "
        )
    raise ValueError(family)


def schedule_reference(data):
    """Enumerate feasible schedules, pruning overlap, budget and unmet dependencies.

    Sorting makes every subset have one unique traversal. Count is computed
    independently of the secondary tie breaks; zero-valued jobs still count.
    Generated instances group overlapping alternatives to bound enumeration.
    """
    level = data['level']
    jobs = sorted(data['jobs'], key=lambda j: (j['start'], j['id']))
    indexes = {job['id']: i for i, job in enumerate(jobs)}
    required = [sum(1 << indexes[key] for key in set(job.get('requires', []))) for job in jobs]
    best_value, best_ids, count = 0, [], 0

    def visit(after, last, remaining, mask, value, ids):
        nonlocal best_value, best_ids, count
        if value > best_value:
            best_value, best_ids, count = value, list(ids), 1
        elif value == best_value:
            count = (count + 1) % 1_000_000_007
            if (len(ids), ids) < (len(best_ids), best_ids):
                best_ids = list(ids)
        for i in range(after, len(jobs)):
            job = jobs[i]
            cost = job.get('cost', 0) if level >= 2 else 0
            if cost > remaining or (level == 3 and required[i] & mask != required[i]):
                continue
            cooldown = data['cooldown'][jobs[last]['kind']][job['kind']] if level >= 2 and last >= 0 else 0
            if last >= 0 and jobs[last]['end'] + cooldown > job['start']:
                continue
            visit(i + 1, i, remaining - cost, mask | (1 << i), value + job['value'], ids + [job['id']])

    visit(0, -1, data.get('budget', 0) if level >= 2 else 0, 0, 0, [])
    result = dict(value=best_value, jobs=best_ids)
    if level == 3:
        result['count'] = count
    return result


def kv_reference(data):
    """Snapshot-based oracle; tests also use an independent undo-log interpreter."""
    state, stack, result = {}, [], []
    for time, action, *args in data['operations']:
        def expire():
            for key, (_, deadline) in list(state.items()):
                if deadline is not None and deadline <= time:
                    del state[key]
        expire()
        value = None
        if action == 'begin':
            stack.append(state.copy())
        elif action == 'commit':
            stack.pop()
        elif action == 'rollback':
            state = stack.pop()
            expire()
        elif action == 'get':
            value = state.get(args[0], (None, None))[0]
        elif action == 'scan':
            value = [[k, state[k][0]] for k in sorted(state) if k.startswith(args[0])]
        elif action == 'delete':
            state.pop(args[0], None)
        else:
            key = args[0]
            write = action == 'put' or state.get(key, (None, None))[0] == args[1]
            if action == 'cas':
                value = write
            if write:
                new, ttl = args[-2:]
                state[key] = (new, None if ttl is None else time + ttl)
                expire()
        result.append(value)
    return result


def make_cases(family, level, split, seed, task_index, count, size):
    version = 2 if family == 'project_schedule' else 1
    rng = random.Random(f'challenge-v{version}:{family}:{level}:{split}:{seed}:{task_index}')
    cases = []
    for index in range(count):
        tag = f'{split}:{seed}:{task_index}:{index}:'
        if family == 'project_schedule':
            jobs = []
            width = 3 + index % 3
            for i in range(size):
                start = (i // width) * 4 + rng.randrange(2)
                end, value = start + rng.choice((3, 4, 7, 11)), rng.randint(-2, 9)
                if i < 3:
                    # A long job enclosing two compatible short jobs rejects a
                    # start-sorted prefix DP that assumes end times are sorted.
                    start, end, value = ((0, 12, 10), (1, 2, 1), (3, 4, 9))[i]
                job = dict(id=tag + f'job-{i:02}', start=start, end=end, value=value)
                if level >= 2:
                    job.update(cost=rng.randint(0, 4), kind=rng.randrange(3))
                if level == 3:
                    earlier = [j['id'] for j in jobs if j['end'] <= start]
                    job['requires'] = rng.sample(earlier, min(len(earlier), rng.choice((0, 0, 1, 2))))
                jobs.append(job)
            # Most cases have a positive feasible schedule; also test zero-value
            # ties and empty input. Every case has a valid optimal answer.
            if level >= 2:
                jobs[0]['cost'] = 0
            rng.shuffle(jobs)
            data = dict(level=level, jobs=jobs)
            if level >= 2:
                data.update(budget=4 + index % 5,
                            cooldown=[[rng.randrange(3) for _ in range(3)] for _ in range(3)])
            if index % 8 == 6:
                for job in jobs:
                    job['value'] = 0
            elif index % 8 == 7:
                data['jobs'] = []
            answer = schedule_reference(data)
        elif family == 'transactional_kv':
            a, b, c = [tag + key for key in ('a', 'ab', 'b')]
            operations = [[0, 'put', a, 7, None], [1, 'begin'], [2, 'put', a, 9, None],
                          [3, 'get', a], [4, 'rollback'], [5, 'get', a], [5, 'delete', a], [5, 'get', a]]
            if level >= 2:
                operations += [[6, 'put', a, 7, 3], [7, 'begin'], [8, 'put', a, 9, None],
                               [9, 'rollback'], [9, 'get', a], [9, 'cas', a, None, 5, 0],
                               [9, 'get', a], [9, 'cas', a, 5, 6, None]]
            if level == 3:
                operations += [[10, 'put', a, 1, None], [11, 'begin'], [12, 'put', a, 2, None],
                               [13, 'begin'], [14, 'put', a, 3, None], [15, 'commit'], [16, 'rollback'],
                               [17, 'put', b, -1, None], [17, 'scan', a], [17, 'get', a]]
            time, depth = 20, 0
            max_depth = 3 if level == 3 else 1
            for _ in range(3 * size):
                time += rng.randrange(3)
                choices = ['put', 'get', 'get', 'delete']
                if depth < max_depth:
                    choices.append('begin')
                if depth:
                    choices += ['commit', 'rollback']
                if level >= 2:
                    choices += ['cas']
                if level == 3:
                    choices += ['scan']
                action, key = rng.choice(choices), rng.choice((a, b, c))
                op = [time, action]
                if action == 'begin':
                    depth += 1
                elif action in ('commit', 'rollback'):
                    depth -= 1
                elif action in ('put', 'cas'):
                    op.append(key)
                    if action == 'cas':
                        op.append(rng.choice((None, -1, 0, 1, 5)))
                    op += [rng.randint(-5, 9), None if level == 1 else rng.choice((None, 0, 1, 3, 8))]
                elif action == 'scan':
                    op.append(rng.choice((tag, a, tag + 'missing')))
                else:
                    op.append(key)
                operations.append(op)
            data = dict(level=level, operations=operations)
            if index % 8 == 7:
                data['operations'] = []
            answer = kv_reference(data)
        else:
            raise ValueError(family)
        cases.append((data, answer))
    return cases
