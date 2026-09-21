import urllib.request, json

def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())

# Test datasets endpoint
ds = get('http://localhost:8000/api/v1/datasets')
print('Datasets:', len(ds), 'found')
for d in ds:
    print(' ', d['id'], d['name'][:40], 'is_default='+str(d['is_default']), 'status='+d['status'], 'accounts='+str(d['record_count']))

# Test dataset status
if ds:
    s = get('http://localhost:8000/api/v1/datasets/'+str(ds[0]['id'])+'/status')
    print('Status:', s['status'], 'stage='+str(s.get('stage')))

# Test accounts with dataset_id
acc = get('http://localhost:8000/api/v1/accounts?dataset_id='+str(ds[0]['id'])+'&page=1&page_size=3&sort_by=bot_probability&sort_order=desc')
print('Accounts: total='+str(acc['total']), 'dataset_id='+str(acc['dataset_id']))

# Test communities with dataset_id
comms = get('http://localhost:8000/api/v1/communities?dataset_id='+str(ds[0]['id']))
print('Communities:', len(comms), 'found')

# Test coordination with dataset_id
clusters = get('http://localhost:8000/api/v1/coordination/clusters?dataset_id='+str(ds[0]['id']))
print('Coordination clusters:', len(clusters))

print('ALL ENDPOINTS OK')
