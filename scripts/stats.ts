import { MongoClient } from 'mongodb';

if (!Bun.env.DB_URI) {
  throw new Error('DB_URI environment variable is required');
}

// MongoDB setup
const client = new MongoClient(Bun.env.DB_URI);

(async () => {
  try {
    await client.connect();
    console.log('Connected to MongoDB');
    const db = client.db();

    // Get all race sessions
    const sessions = await db.collection('race_sessions').find().toArray();
    const TOTAL_SESSIONS = sessions.length;

    // Forge Usage Stats
    const uniqueUsers = new Set(sessions.map((session) => session.address)).size;

    // Challenge popularity
    const challengeCounts: Record<string, number> = sessions.reduce(
      (acc, session) => {
        acc[session.challenge] = (acc[session.challenge] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>
    );

    const sortedChallenges = Object.entries(challengeCounts)
      .sort(([, a], [, b]) => (b as number) - (a as number))
      .map(([challenge, count]) => `${challenge}: ${count}`);

    // Demonstration Data Stats
    let textPrompts = 0;
    let videoMinutes = 0;
    let keyboardEvents = 0;
    let mouseEvents = 0;
    let rewardEvents = 0;
    let totalRewards = 0;

    // Get all training events
    const trainingEvents = await db.collection('training_events').find().toArray();

    // Process training events
    trainingEvents.forEach((event) => {
      if (event.type === 'reward') {
        rewardEvents++;
        if (event.metadata?.rewardValue) {
          totalRewards += event.metadata.rewardValue;
        }
      }
    });

    textPrompts = trainingEvents.length;

    // Calculate video duration from sessions
    sessions.forEach((session) => {
      const start = new Date(session.created_at).getTime();
      const end = new Date(session.updated_at).getTime();
      videoMinutes += (end - start) / (1000 * 60);
    });

    // Process guac files for input events
    for (const session of sessions) {
      const guacFile = `./data/${session._id}.guac`;
      if (await Bun.file(guacFile).exists()) {
        try {
          // Read file in chunks to handle large files
          const content = await Bun.file(guacFile).text();
          const instructions = content.split(';');

          instructions.forEach((instruction) => {
            if (!instruction.trim()) return;
            const parts = instruction.trim().split(',');
            if (parts.length === 0) return;

            // Simple opcode extraction
            const opMatch = parts[0].match(/^\d+\.(\w+)/);
            if (!opMatch) return;
            const opcode = opMatch[1];

            if (opcode === 'key') keyboardEvents++;
            else if (opcode === 'mouse') mouseEvents++;
          });
        } catch (e) {
          console.error(`Error processing ${guacFile}:`, e.message);
        }
      }
    }

    console.log('\n=== Forge Usage ===');
    console.log(`Sessions: ${TOTAL_SESSIONS}`);
    console.log(`Unique Users: ${uniqueUsers}`);

    console.log('\nChallenge Distribution:');
    sortedChallenges.forEach((c) => {
      const [name, count] = c.split(':');
      console.log(`${name.trim()}: ${count.trim()}`);
    });

    console.log('\n=== Rewards Data ===');
    console.log(`• ${rewardEvents.toLocaleString()} reward events`);
    console.log(`• ${totalRewards.toLocaleString()} total rewards`);

    console.log('\n=== Demonstration Data ===');
    console.log('Input:');
    console.log(`• ${textPrompts.toLocaleString()} text prompts`);
    console.log(`• ${videoMinutes.toLocaleString()} minutes of video`);

    console.log('\nOutput (User Actions):');
    console.log(`• ${keyboardEvents.toLocaleString()} keyboard events`);
    console.log(`• ${mouseEvents.toLocaleString()} mouse events`);
    console.log(`• ${(keyboardEvents + mouseEvents).toLocaleString()} total user actions`);
  } catch (error) {
    console.error('Error:', error);
  } finally {
    await client.close();
  }
})();
