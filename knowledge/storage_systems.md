# Battery Energy Storage Systems for Solar

## Sizing guidance
A practical battery sizing heuristic is 2 to 4 hours of storage at 50-80 percent of plant peak output. For a 1 MW peak solar plant, 0.5-0.8 MW / 2-4 MWh is a typical utility-scale pairing. [Source: NREL Storage Futures 2021]

## Charge window optimization
Charging is most efficient between 10:00 and 14:00 local time when irradiance is high and the plant would otherwise be exporting excess energy. Avoid charging past 80 percent state-of-charge to extend cycle life. [Source: IRENA Electricity Storage 2017]

## Discharge scheduling
Discharge the battery during the evening peak (17:00-21:00 typically) to shift solar energy to high-demand hours. This captures the highest time-of-use tariff differential and reduces grid stress during the duck curve ramp. [Source: LBNL Grid Energy Storage 2019]

## Round-trip efficiency
Modern lithium-ion BESS achieve 85-92 percent round-trip efficiency. Factor efficiency losses into economic dispatch: a 90 percent RTE means 10 percent of stored solar is lost, which should be priced against direct export revenue. [Source: EIA Battery Storage 2023]

## Cycle life and depth of discharge
Limiting depth of discharge to 80 percent roughly doubles cycle life compared to 100 percent DoD. Over a 10-year horizon this can reduce LCOE of storage by 15-25 percent. [Source: Sandia Battery Testing Manual]

## State of charge buffer for forecast error
Maintain a 10-15 percent SoC buffer against next-day solar forecast uncertainty to ensure discharge commitments can be met on low-irradiance days. [Source: DOE Grid Storage Handbook]
