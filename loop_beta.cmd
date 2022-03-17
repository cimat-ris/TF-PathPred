$i = 0;

while ($i -lt 4)
{
    $j = 0;
	while ($j -lt 4)
	{
		python beta_experiment.py --dataset %i --beta %j
		$j++	
	}
	$i++
}