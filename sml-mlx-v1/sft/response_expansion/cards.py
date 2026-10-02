"""Original task cards; fixed topic partitions, no external model or corpus."""

# (question, complete target, concept alternatives). Topics never cross splits.
EXPLANATIONS = {
 'train': [
  ('Why does the Moon look bright?', 'The Moon looks bright because it reflects light from the Sun.', [['reflects', 'reflect'], ['sun', 'sunlight']]),
  ('Why does ice melt when warmed?', 'Ice melts because added heat changes it from a solid into liquid water.', [['heat', 'warm'], ['liquid', 'water']]),
  ('Why do roots help a plant?', 'Roots absorb water and nutrients from the soil and anchor the plant.', [['water'], ['nutrients'], ['soil']]),
  ('Why do birds have feathers?', 'Feathers help birds stay warm, and flight feathers help many birds fly.', [['warm', 'insulation'], ['fly', 'flight']]),
  ('Why can we see through a clear window?', 'We can see through a clear window because its glass lets light pass through.', [['light'], ['pass', 'passes', 'transmits']]),
  ('Why do leaves move on a windy day?', 'Leaves move because the moving air pushes against them.', [['air', 'wind'], ['pushes', 'push', 'force']]),
  ('Why does a dropped stone fall?', 'A dropped stone falls because Earth\'s gravity pulls it downward.', [['gravity'], ['pulls', 'pull', 'downward']]),
  ('Why does a metal spoon get hot in hot soup?', 'Heat travels from the hot soup through the metal spoon by conduction.', [['heat'], ['conduction', 'conducts', 'travels']]),
  ('Why do shoes have soles?', 'Soles protect the feet from rough surfaces and provide grip on the ground.', [['protect'], ['feet'], ['grip', 'traction']]),
  ('Why does a book have an index?', 'An index helps readers find the pages where particular topics are discussed.', [['find', 'locate'], ['pages'], ['topics', 'subjects']]),
  ('Why can a magnet pick up an iron nail?', 'A magnet can pick up an iron nail because magnetic force attracts the iron.', [['magnetic', 'magnet'], ['attracts', 'attract'], ['iron']]),
  ('Why is soap useful for washing greasy dishes?', 'Soap helps grease mix with water so the grease can be rinsed away.', [['grease'], ['water'], ['rinsed', 'rinse', 'washed', 'wash']]),
  ('Why do flowers attract bees?', 'Many flowers provide nectar and pollen, which bees collect for food.', [['nectar'], ['pollen'], ['food']]),
  ('Why does a door have hinges?', 'Hinges let a door swing open and closed while staying attached to its frame.', [['swing', 'rotate'], ['attached', 'frame']]),
  ('Why do we use a map?', 'A map shows where places are and helps people plan a route between them.', [['places', 'locations'], ['route', 'directions']]),
  ('Why does bread rise when yeast is added?', 'Yeast produces carbon dioxide gas, which forms bubbles that make the dough rise.', [['yeast'], ['carbon dioxide'], ['gas', 'bubbles']]),
 ],
 'dev': [
  ('Why is wool used in warm clothing?', 'Wool traps air that slows heat loss from the body.', [['air'], ['heat'], ['traps', 'trap', 'slows', 'insulates']]),
  ('Why does a bicycle need brakes?', 'Brakes create friction that slows the wheels and helps the bicycle stop.', [['friction'], ['slow', 'slows', 'stop']]),
  ('Why does an echo happen?', 'An echo happens when sound reflects from a surface and returns to the listener.', [['sound'], ['reflects', 'reflection', 'bounces']]),
  ('Why do seeds need water to germinate?', 'Water activates processes in a seed that allow it to begin growing.', [['water'], ['grow', 'growing', 'growth'], ['activates', 'begin', 'start']]),
 ],
 'test': [
  ('Why does a puddle gradually disappear?', 'Water in the puddle evaporates into the air as water vapor.', [['water'], ['evaporates', 'evaporation'], ['air', 'vapor']]),
  ('Why do ducks have webbed feet?', 'Webbed feet help ducks push against water when they swim.', [['water'], ['push', 'paddle'], ['swim', 'swimming']]),
  ('Why does a compass point north?', 'A compass needle aligns with Earth\'s magnetic field.', [['magnetic field'], ['aligns', 'align', 'needle']]),
  ('Why do curtains make a room darker?', 'Curtains block some of the light that would otherwise enter the room.', [['light'], ['block', 'blocks', 'blocking']]),
 ]}

# Topic-separated grounded causal questions. The reason is supplied in the input.
CAUSES = {
 'train': [
  ('The ferry trip was canceled', 'strong winds made the crossing unsafe'),
  ('The bakery opened late', 'its oven needed repairs'),
  ('The path was closed', 'a fallen branch blocked it'),
  ('The outdoor concert moved inside', 'heavy rain was expected'),
  ('The library extended its hours', 'students needed a quiet place to study'),
  ('The gardener watered the seedlings', 'the soil was dry'),
  ('The shop ordered more notebooks', 'its notebook shelves were empty'),
  ('The delivery was delayed', 'the vehicle had a flat tire'),
  ('The class used a different room', 'the usual room was being painted'),
  ('The museum covered the painting', 'workers nearby were making dust'),
  ('The hikers turned back', 'thick fog hid the route'),
  ('The kitchen window was opened', 'smoke from burnt toast filled the room'),
  ('The visitors spoke softly', 'a recording was taking place'),
  ('The volunteer replaced the sign', 'the old lettering had faded'),
  ('The dog waited by the gate', 'its owner had gone through the gate'),
  ('The park added benches', 'visitors needed more places to rest'),
 ],
 'dev': [
  ('The curtains were drawn', 'bright sunlight was falling on the screen'),
  ('The football game was postponed', 'the field was flooded'),
  ('The florist moved the pots indoors', 'frost was forecast'),
  ('The workshop borrowed a lamp', 'its workbench was poorly lit'),
 ],
 'test': [
  ('The train stopped outside the station', 'a signal ahead was red'),
  ('The rehearsal ended early', 'the hall had to close for repairs'),
  ('The picnic used a sheltered table', 'a strong breeze kept lifting the napkins'),
  ('The visitor returned to the counter', 'a bag had been left there'),
 ]}

POLITE = {
 'train': ['open the gate','bring the notebook','move the chair','lower your voice',
           'leave the light on','hold the ladder','read the notice','return the key',
           'wait by the entrance','carry the basket','put the cup on the tray','check the address',
           'send the photograph','water the flowers','wipe the table','show me the way'],
 'dev': ['sweep the floor','fold the blanket','close the drawer','fetch the cushion'],
 'test': ['hang up the coat','rinse the bowl','switch off the fan','hand me the towel']}

CLARIFY = {
 'train': [
  ('I need a bag but cannot choose one.', 'What do you need to carry in the bag?', [['carry']]),
  ('I want to visit a new place but have not decided where.', 'What kinds of places do you enjoy visiting?', [['places'], ['enjoy', 'like']]),
  ('I am looking for a book and need a suggestion.', 'What kinds of books do you enjoy reading?', [['books'], ['enjoy', 'like']]),
  ('I need a coat and am unsure which to choose.', 'What weather will you wear the coat in?', [['weather']]),
  ('I would like to start a hobby but cannot decide.', 'What activities do you already enjoy?', [['activities'], ['enjoy', 'like']]),
  ('I need a present and do not know what to get.', 'Who is the present for?', [['who']]),
  ('I want to decorate a room and need some ideas.', 'What style would you like the room to have?', [['style']]),
  ('I need a recipe but have not chosen a dish.', 'What ingredients do you have available?', [['ingredients']]),
 ],
 'dev': [
  ('I need a plant but do not know which kind to buy.', 'How much sunlight will the plant receive?', [['sunlight', 'light']]),
  ('I want to listen to music and need a suggestion.', 'What kinds of music do you usually enjoy?', [['music'], ['enjoy', 'like']]),
  ('I need a place to eat and cannot decide.', 'What kind of food would you like?', [['food', 'cuisine']]),
  ('I want to take a class but have not picked a subject.', 'What would you like to learn?', [['learn']]),
 ],
 'test': [
  ('I need walking shoes and am unsure which to buy.', 'What kind of terrain will you walk on?', [['terrain', 'surfaces', 'paths']]),
  ('I want to watch a film and cannot choose.', 'What kinds of films do you enjoy?', [['films', 'movies'], ['enjoy', 'like']]),
  ('I need a desk but have not chosen one.', 'How much space do you have for the desk?', [['space']]),
  ('I want a pet but have not decided which kind.', 'How much time can you spend caring for a pet?', [['time'], ['care', 'caring']]),
 ]}

GREETINGS = {
 'train': ['Hello!','Hi!','Good morning!','Good afternoon!','Good evening!','Hey!',
           'Hello, can you help?','Hi, I have a question.','Good day!','Hello there!',
           'Hi, are you ready to help?','Hey, I would like to ask something.',
           'Hello, nice to meet you.','Hi, could we talk?','Greetings!',
           'Hello, I need a little help.','Hi, may I ask a question?',
           'Hey, can we get started?','Hello, I am glad to be here.',
           'Hi, I would appreciate some help.','Hello, are you available?',
           'Hey, I have something to ask.','Hello, shall we begin?',
           'Hi, I could use some guidance.','Hello, I would like to chat.',
           'Hi, thanks for being here.','Hey, can I ask you something?',
           'Hello, please help me get started.','Hi, I need to ask about something.',
           'Hello, I am looking for assistance.','Hey, nice to meet you.',
           'Hi, let us start a conversation.'],
 'dev': ['Morning!','Evening!','Hello, could you assist me?', 'Hi, is this a good time to ask?',
         'Good afternoon, I have something to discuss.','Hello, I hope you can help me.',
         'Hey, shall we talk?','Hi, I would like your assistance.'],
 'test': ['Good morning, may I ask something?','Hello, I am ready to begin.',
          'Hey there, I need some guidance.','Hi, could I get your help?',
          'Good evening, can we chat?','Hello, it is nice to be here.',
          'Hi, I have a topic to discuss.','Greetings, can you assist?',
          'Hey, I would appreciate your help.','Hello, may we get started?',
          'Hi, I am hoping for some guidance.','Good day, could we talk?']}
