import os
i=0

regions = ['MidTown','CenterPark','WallStreet']
days = ['weekdays','weekends']
for day in days:
    for region in regions:
        commands = ["python optimization_DSML_main.py "+str(region) +" Product withMin withoutQhat 99999 99999 15 Nestgra 0.1 0.05 3.5 12 200 withMandist "+str(day),
                    "python optimization_DSML_main.py "+str(region) +" Product withMin withoutQhat 99999 99999 5 Nestgra 0.1 0.05 3.5 12 200 withMandist "+str(day),
                    "python optimization_DSML_main.py "+str(region) +" Product withMin withoutQhat 99999 99999 10 Nestgra 0.1 0.05 3.5 12 200 withMandist "+str(day),
                    "python optimization_DSML_main.py "+str(region) +" Product withMin withoutQhat 99999 99999 20 Nestgra 0.1 0.05 3.5 12 200 withMandist "+str(day),
                    "python optimization_DSML_main.py " + str(region) + " Product withMin withoutQhat 99999 99999 25 Nestgra 0.1 0.05 3.5 12 200 withMandist "+str(day),
                    "python optimization_DSML_main.py " + str(region) + " Product withMin withoutQhat 99999 99999 30 Nestgra 0.1 0.05 3.5 12 200 withMandist "+str(day),
    
                    "python optimization_DSML_main.py " + str(region) + " Product withMin withoutQhat 0 100 15 Nestgra 0.1 0.05 3.5 12 200 withMandist "+str(day),
                    "python optimization_DSML_main.py "+str(region) +" Product withMin withoutQhat 0 5 15 Nestgra 0.1 0.05 3.5 12 200 withMandist "+str(day),
                    "python optimization_DSML_main.py "+str(region) +" Product withMin withoutQhat 0 3 15 Nestgra 0.1 0.05 3.5 12 200 withMandist "+str(day),
                    "python optimization_DSML_main.py "+str(region) +" Product withMin withoutQhat 0 2 15 Nestgra 0.1 0.05 3.5 12 200 withMandist "+str(day),
                    "python optimization_DSML_main.py "+str(region) +" Product withMin withoutQhat 0 2.5 15 Nestgra 0.1 0.05 3.5 12 200 withMandist "+str(day),
                    "python optimization_DSML_main.py "+str(region) +" Product withMin withoutQhat 0 2.4 15 Nestgra 0.1 0.05 3.5 12 200 withMandist "+str(day),
                    "python optimization_DSML_main.py "+str(region) +" Product withMin withoutQhat 0 2.3 15 Nestgra 0.1 0.05 3.5 12 200 withMandist "+str(day),
                    "python optimization_DSML_main.py "+str(region) +" Product withMin withoutQhat 0 2.2 15 Nestgra 0.1 0.05 3.5 12 200 withMandist "+str(day),
                    "python optimization_DSML_main.py "+str(region) +" Product withMin withoutQhat 0 2.1 15 Nestgra 0.1 0.05 3.5 12 200 withMandist "+str(day),
                    "python optimization_DSML_main.py "+str(region) +" Product withMin withoutQhat 0 2.25 15 Nestgra 0.1 0.05 3.5 12 200 withMandist "+str(day),

                    "python optimization_DSML_main.py " + str(region) + " Product withMin withoutQhat 0 1.9 15 Nestgra 0.1 0.05 3.5 12 200 withMandist " + str(day),
                    "python optimization_DSML_main.py " + str(region) + " Product withMin withoutQhat 0 1.8 15 Nestgra 0.1 0.05 3.5 12 200 withMandist " + str(day),
                    "python optimization_DSML_main.py " + str(region) + " Product withMin withoutQhat 0 1.7 15 Nestgra 0.1 0.05 3.5 12 200 withMandist " + str(day),
                    "python optimization_DSML_main.py " + str(region) + " Product withMin withoutQhat 0 1.6 15 Nestgra 0.1 0.05 3.5 12 200 withMandist " + str(day),
                    "python optimization_DSML_main.py " + str(region) + " Product withMin withoutQhat 0 1.5 15 Nestgra 0.1 0.05 3.5 12 200 withMandist " + str(day),
                    "python optimization_DSML_main.py " + str(region) + " Product withMin withoutQhat 0 1.4 15 Nestgra 0.1 0.05 3.5 12 200 withMandist " + str(day),
                    "python optimization_DSML_main.py " + str(region) + " Product withMin withoutQhat 0 1.3 15 Nestgra 0.1 0.05 3.5 12 200 withMandist " + str(day),
                    "python optimization_DSML_main.py " + str(region) + " Product withMin withoutQhat 0 1.2 15 Nestgra 0.1 0.05 3.5 12 200 withMandist " + str(day),
                    "python optimization_DSML_main.py " + str(region) + " Product withMin withoutQhat 0 1.1 15 Nestgra 0.1 0.05 3.5 12 200 withMandist " + str(day),
                    "python optimization_DSML_main.py " + str(region) + " Product withMin withoutQhat 0 1 15 Nestgra 0.1 0.05 3.5 12 200 withMandist " + str(day),
                    "python optimization_DSML_main.py " + str(region) + " Product withMin withoutQhat 0 0.8 15 Nestgra 0.1 0.05 3.5 12 200 withMandist " + str(day)
                    
                    ]
        
        for command in commands:
            print('===============' + str(i + 1) + '===============')
            print(command)
            os.system(command)
            i += 1