> **Probably not needed (2026-10-21):** the official videos.insert reference now says "Videos uploaded from
> unverified API projects are not restricted to private viewing mode." Confirm with one test upload first;
> keep this draft only in case the lock still applies in practice.

# YouTube API Services audit: draft answers

Form: https://support.google.com/youtube/contact/yt_api_form ("Audit and quota extension")
Project: clipper-512318 (OAuth client type: Desktop app, app name "Clip")

## Reason for request
Remove the "private-only" restriction on videos uploaded via videos.insert from an unverified project.
No quota increase needed (expected usage: 1-3 uploads/day, ~5,000 units/day, below the default 10,000).

## API client description
Clip (Clipper Uploader) is a private, single-user command-line tool run by the channel owner on their own computer.
It uploads short-form videos (YouTube Shorts) that the owner has edited and manually approved to the owner's
own YouTube channel. It is not offered to other users, has no public website, and stores no data about
other YouTube users.

## Use case
- Content: clips from livestreams whose creators explicitly permit clipping (paid clipping campaigns, e.g. Whop
  Content Rewards), edited with captions, a hook and a cold open; every clip credits the original creator.
- Every upload is reviewed and approved by the channel owner before it is sent to the API.
- API usage: videos.insert only (scope youtube.upload). Title, description and tags are set by the owner.
- No YouTube data is read, stored, aggregated or shared. The OAuth token is stored locally (file mode 0600)
  and can be revoked at any time from the Google Account settings.

## Expected usage
- Users: 1 (the channel owner)
- Uploads: 1-3 per day
- Quota: default 10,000 units/day is sufficient

## Links required by the form
- Home page: https://mokus2112.github.io/clipper-privacy/
- Privacy policy: https://mokus2112.github.io/clipper-privacy/privacy.html
- Terms of service: https://mokus2112.github.io/clipper-privacy/terms.html
- Source repository: https://github.com/Mokus2112/clipper
- Contact email: adi.bereczky@gmail.com

## Notes
- Answer honestly; do not overstate the scale. Outcome and timing are not guaranteed.
